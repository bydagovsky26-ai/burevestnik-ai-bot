
import os
import json
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not configured")


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("burevestnik")


# ============================================================
# CLIENTS
# ============================================================

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================================
# APPLICATION MEMORY
# Temporary memory for the first version.
# PostgreSQL will replace this later.
# ============================================================

@dataclass
class Application:
    telegram_user_id: int

    messages: list = field(default_factory=list)

    service_type: Optional[str] = None

    origin: Optional[str] = None
    destination: Optional[str] = None

    pickup_address: Optional[str] = None
    delivery_address: Optional[str] = None

    cargo_description: Optional[str] = None
    cargo_article: Optional[str] = None

    weight_kg: Optional[float] = None
    length_cm: Optional[float] = None
    width_cm: Optional[float] = None
    height_cm: Optional[float] = None
    volume_m3: Optional[float] = None
    pieces: Optional[int] = None

    cargo_value: Optional[float] = None
    cargo_currency: Optional[str] = None

    purchase_required: Optional[bool] = None
    inspection_required: Optional[bool] = None
    packing_required: Optional[bool] = None

    documents_available: Optional[bool] = None

    deadline: Optional[str] = None
    pickup_date: Optional[str] = None
    urgency: Optional[str] = None

    payment_method: Optional[str] = None

    complexity: Optional[str] = None

    special_requirements: Optional[str] = None

    status: str = "new"


applications: dict[int, Application] = {}


def get_application(user_id: int) -> Application:
    if user_id not in applications:
        applications[user_id] = Application(
            telegram_user_id=user_id
        )

    return applications[user_id]


# ============================================================
# AI SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
Ты — AI-оператор международной логистической компании «Буревестник».

Твоя задача — профессионально принимать и разбирать заявки клиентов
на доставку, выкуп, поиск, проверку и другие логистические услуги.

Ты НЕ являешься простым чат-ботом.

Твоя главная задача:
1. Понять, что клиент хочет сделать.
2. Извлечь уже известную информацию.
3. Определить, каких критически важных данных не хватает.
4. Задать следующий наиболее полезный вопрос.
5. После каждого ответа заново оценивать заявку.
6. Никогда не задавать клиенту длинную анкету без необходимости.

ОБЯЗАТЕЛЬНО УТОЧНЯЙ СРОКИ.

Сроки критически важны, потому что от них зависят:
- возможность выполнить заявку;
- доступный транспорт;
- маршрут;
- стоимость;
- срочный тариф.

Если клиент говорит:
«срочно», «на днях», «как можно быстрее», «на следующей неделе»,
нужно получить максимально понятный срок.

Если точная дата пока неизвестна, можно использовать понятное
описание срочности и позже уточнить.

------------------------------------------------------------
ГРУЗ
------------------------------------------------------------

Нужно учитывать:
- что именно перевозится;
- количество;
- вес;
- размеры;
- объём;
- стоимость;
- новое или б/у;
- упаковку;
- хрупкость;
- наличие жидкостей;
- масла;
- духи;
- аэрозоли;
- аккумуляторы;
- батареи;
- химические вещества;
- краски;
- технические жидкости;
- другие потенциально опасные или ограниченные категории.

Никогда не делай вывод только по одному слову.

Например, если клиент говорит:
«духи», «масло», «краска», «батарея»,
нужно понять точный товар, количество, упаковку и другие необходимые
характеристики перед окончательным выводом о возможности перевозки.

------------------------------------------------------------
ДОКУМЕНТЫ
------------------------------------------------------------

Не считай автоматически:
«нет документов = перевозка невозможна».

Сначала определи:
- какой груз;
- какое направление;
- какой способ перевозки;
- какие документы имеются;
- какие документы могут потребоваться.

Если информации недостаточно — сообщи, что нужна дополнительная проверка.

------------------------------------------------------------
СТОИМОСТЬ ГРУЗА
------------------------------------------------------------

Если клиент называет стоимость, не считай её автоматически правильной.

Если есть признаки несоответствия между заявленной стоимостью,
моделью, артикулом или описанием товара, нужно запросить:
- артикул;
- модель;
- ссылку;
- фото;
- документы;
или другую информацию для проверки.

Стоимость груза может влиять на:
- риск;
- ответственность;
- страхование;
- таможенные вопросы;
- стоимость услуги.

------------------------------------------------------------
СЛОЖНОСТЬ РАБОТЫ
------------------------------------------------------------

Стоимость услуги зависит не только от веса.

Учитывай:
- количество точек;
- расстояние до места забора;
- магазин / склад / частный адрес;
- необходимость найти товар;
- артикул;
- покупку;
- переговоры с продавцом;
- проверку товара;
- фото/видео;
- упаковку;
- переупаковку;
- погрузку;
- этаж;
- лифт;
- парковку;
- пропускной режим;
- документы;
- доверенность;
- срочность;
- работу ночью/в выходные;
- особые условия.

Простая доставка и сложная представительская работа
не должны оцениваться одинаково.

------------------------------------------------------------
ПОКУПКА И ПРОВЕРКА
------------------------------------------------------------

Если клиент просит:
- найти товар;
- купить товар;
- встретиться с продавцом;
- проверить товар;
- сделать фотографии;
- снять видео;
- проверить артикул;
- упаковать;
- отвезти перевозчику;

это отдельные операции и должны учитываться при расчёте стоимости.

Цена самого товара и стоимость работы «Буревестника» — разные вещи.

------------------------------------------------------------
МАРШРУТ
------------------------------------------------------------

В будущем система будет использовать:
- авиа;
- автомобильную перевозку;
- море;
- железную дорогу;
- автобус;
- мультимодальные маршруты.

При выборе маршрута нужно учитывать одновременно:
- срок;
- стоимость;
- возможность перевозки;
- ограничения груза;
- риск;
- доступность транспорта.

Нельзя придумывать рейсы, тарифы или сроки.

------------------------------------------------------------
ОПЛАТА
------------------------------------------------------------

Компания может работать с различными способами оплаты, включая:
- наличные;
- безналичный расчёт;
- договор;
- счёт;
- криптовалюту;
- личную встречу;
- другие согласованные варианты.

Если способ оплаты влияет на стоимость или условия,
это должно учитываться.

------------------------------------------------------------
ЦЕНА
------------------------------------------------------------

AI не должен придумывать окончательную цену,
если нет подтверждённых данных.

В будущем цена будет рассчитываться примерно как:

транспорт
+ забор
+ доставка
+ упаковка
+ покупка
+ проверка
+ представительская работа
+ сложность
+ срочность
+ дополнительные расходы
+ риск
+ комиссии
+ маржа компании.

------------------------------------------------------------
СТИЛЬ ОБЩЕНИЯ
------------------------------------------------------------

Общайся по-русски.

Будь:
- коротким;
- понятным;
- профессиональным;
- человеческим.

Не задавай десять вопросов сразу.

Если для следующего шага нужен только один параметр —
задай один вопрос.

Если клиент уже дал информацию — никогда не спрашивай её повторно.

Если данных достаточно для предварительной оценки —
дай предварительную оценку и обозначь, что нужно подтвердить.

Если выполнить заявку невозможно —
объясни причину и, если возможно, предложи альтернативу.

Никогда не выдумывай факты.
"""


# ============================================================
# APPLICATION SERIALIZATION
# ============================================================

def application_to_dict(app: Application) -> dict:
    return {
        "service_type": app.service_type,
        "origin": app.origin,
        "destination": app.destination,
        "pickup_address": app.pickup_address,
        "delivery_address": app.delivery_address,
        "cargo_description": app.cargo_description,
        "cargo_article": app.cargo_article,
        "weight_kg": app.weight_kg,
        "length_cm": app.length_cm,
        "width_cm": app.width_cm,
        "height_cm": app.height_cm,
        "volume_m3": app.volume_m3,
        "pieces": app.pieces,
        "cargo_value": app.cargo_value,
        "cargo_currency": app.cargo_currency,
        "purchase_required": app.purchase_required,
        "inspection_required": app.inspection_required,
        "packing_required": app.packing_required,
        "documents_available": app.documents_available,
        "deadline": app.deadline,
        "pickup_date": app.pickup_date,
        "urgency": app.urgency,
        "payment_method": app.payment_method,
        "complexity": app.complexity,
        "special_requirements": app.special_requirements,
        "status": app.status,
    }


# ============================================================
# AI PROCESSING
# ============================================================

def ask_ai(app: Application, user_message: str) -> str:

    app.messages.append({
        "role": "user",
        "content": user_message
    })

    application_data = json.dumps(
        application_to_dict(app),
        ensure_ascii=False,
        indent=2
    )

    context = f"""
ТЕКУЩАЯ ЗАЯВКА:

{application_data}

ИСТОРИЯ ДИАЛОГА:

{json.dumps(app.messages[-12:], ensure_ascii=False, indent=2)}

ПОСЛЕДНЕЕ СООБЩЕНИЕ КЛИЕНТА:

{user_message}

Проанализируй заявку.

Сначала мысленно определи:
- что клиент хочет;
- что уже известно;
- какие данные критичны;
- какой следующий вопрос даст максимальную пользу.

Не показывай клиенту внутренний анализ.

Если критически важной информации не хватает —
задай только следующий необходимый вопрос.

Если информации достаточно для предварительного ответа —
ответь клиенту.

Не придумывай цены, расписания, рейсы или разрешения.
"""

    response = openai_client.responses.create(
        model="gpt-5.6",
        instructions=SYSTEM_PROMPT,
        input=context
    )

    answer = response.output_text.strip()

    app.messages.append({
        "role": "assistant",
        "content": answer
    })

    return answer


# ============================================================
# TELEGRAM HANDLERS
# ============================================================

@dp.message(CommandStart())
async def start_handler(message: Message):

    app = get_application(message.from_user.id)

    app.status = "new"

    await message.answer(
        "Здравствуйте! 👋\n"
        "Компания «Буревестник».\n\n"
        "Расскажите, что нужно доставить, откуда и куда."
    )


@dp.message(F.text)
async def text_handler(message: Message):

    user_id = message.from_user.id
    app = get_application(user_id)

    try:
        answer = await asyncio.to_thread(
            ask_ai,
            app,
            message.text
        )

        await message.answer(answer)

    except Exception as e:
        logger.exception("AI error: %s", e)

        await message.answer(
            "Не смог обработать заявку. "
            "Попробуйте отправить сообщение ещё раз."
        )


@dp.message()
async def unsupported_handler(message: Message):

    await message.answer(
        "Пока я обрабатываю текстовые сообщения. "
        "Поддержку фото, документов и голосовых подключим следующим модулем."
    )


# ============================================================
# MAIN
# ============================================================

async def main():

    logger.info("Burevestnik AI bot is starting...")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
