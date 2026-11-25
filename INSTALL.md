# Установка Ar3DRA Hub

Как развернуть проект на новом компьютере с нуля. Проект объединяет **UniMate**
(text-to-motion, Python 3.10) и **студию Ar3DRAHub**
(картинка/фраза → 3D → риг → анимация, Python 3.11).

```text
фраза / картинка → 3D-модель → Finish → Rig → UniMate motion → анимированный GLB
```

## 1. Требования

| Что | Требование |
|---|---|
| ОС | Windows или Linux (**macOS не поддерживается** — UniMate его не поддерживает) |
| GPU | NVIDIA (CUDA), драйвер **575+** (иначе Pixal3D придётся компилировать) |
| Python | **3.10** — только для окружения UniMate (ставится отдельно, в `.venv`). Для студии Python 3.11 ставит сам `uv` |
| Диск | ~30 ГБ свободного места под все веса |
| Интернет | Нужен при установке и при первом запуске UniMate (подтягивает текстовый энкодер `flan-t5-base`) |
| Дополнительно | `uv` (установщик студии предложит поставить сам), `git` (для Linux-установщика и `git clone`) |

Для справки: на машине, где проект разрабатывается, тяжёлые папки занимают ~50 ГБ
(`models/` ~27 ГБ, `.venv` ~9 ГБ, `vendor/` ~7 ГБ, `.venv-studio` ~3,5 ГБ,
`output/` ~3 ГБ, `outputs/` ~1,3 ГБ). Всё это пересоздаётся установщиками —
копировать их не нужно.

## 2. Получить код

**Вариант А — скопировать папку.** Копируйте проект целиком, **без** папок, которые
всё равно пересоздаются:

- **исключить:** `.venv`, `.venv-studio`, `outputs/`, `output/`, `models/`, `vendor/`, `dataset/`
- **нужно:** весь остальной код + установщики (`setup.bat`, `install.bat`, `install.sh`,
  `install_studio.bat`, `install_studio.ps1`, `install_studio.sh`, `lab`, `lab.cmd`,
  `run.bat`, `run_inference.bat`, `run_mcp.bat`)

**Вариант Б — git clone:**

```bash
git clone https://github.com/SimarglV/Ar3DRAHub.git
```

## 3. Установка

Два окружения, потому что они используют разные версии numpy:

| Окружение | Python | Что в нём работает | Установщик |
|---|---|---|---|
| `.venv` | 3.10 | UniMate: сэмплинг движения, `rig_preprocess`, анимация меша | `install.bat` / `install.sh` — скачивает чекпойнт (~6 ГБ) |
| `.venv-studio` | 3.11 | Студия, движки, риг, MCP | `install_studio.ps1` / `install_studio.sh` — только пакеты |

### Windows — всё одним кликом

Двойной клик по **`setup.bat`** (или `setup.bat /y` для скриптов и агентов).
Он спросит один раз, что будет установлено, и запустит по очереди оба установщика:
`.venv` с чекпойнтом (~6 ГБ) и `.venv-studio` с пакетами. Никакие другие веса он не качает.

### Windows — по шагам

```powershell
install.bat                       # UniMate: .venv + чекпойнт (~6 ГБ, ~9 ГБ на диске)
install_studio.bat                # студия: .venv-studio (обёртка над install_studio.ps1)
.\lab                             # открыть студию в браузере (run.bat — то же самое)
```

`install_studio.bat` — это двойной клик по `powershell -ExecutionPolicy ByPass -File
install_studio.ps1`: Python 3.11 + пакеты студии, без весов. `uv` ставится сам, если
его нет (спросит).

### Linux

```bash
bash install.sh
bash install_studio.sh
./lab
```

## 4. Первый запуск

В студии (`http://127.0.0.1:8777`):

1. **Setup & Status** → нажать **Set up** у **Pixal3D** (~8.4 ГБ, спросит размер и лицензию).
   TRELLIS.2 и Hunyuan3D-2.1 — только Linux + NVIDIA.
2. Поставить **Blender 4.2+** самостоятельно (нужен для Finish — низкополигональной
   чистки и Pixel Match). Карточка Blender на той же странице покажет, найден ли он.
   На Linux без Blender его можно поставить кнопкой на той же странице
   (`scripts/bootstrap_blender.py` ставит Blender 4.2 LTS в `~/blender-lts/`).
3. Карточка **UniMate** уже должна быть зелёной (✓ ready) — значит чекпойнт из шага 3
   на месте.
4. (Необязательно) **Marigold V2** — для де-лайта текстуры в Finish. Карточка на той же
   странице ставит своё окружение + веса (~22 ГБ, NVIDIA/CUDA) и спрашивает согласие.
   Из командной строки: `python scripts/bootstrap_marigold_v2.py --yes`. Без неё Finish
   просто пропускает де-лайт; нормаль-карта от Marigold включается отдельно.

Дальше: **+ Create** → «Start from: An idea» → «I want: A character moved by text» →
опишите персонажа и напишите промпт движения (например `An object walks forward.`) →
**Make it**.

## 5. Где хранятся модели

По умолчанию — всё внутри проекта:

```
models/
├── huggingface/   ← все HF-модели (TRELLIS.2, Hunyuan3D, Qwen, …)
├── u2net/         ← модель вырезания фона
└── <движок>/      ← Pixal3D и другие
```

Чекпойнт UniMate — в `outputs/unimate_uniml3d_f60_v3/` (его ставит `install.bat`).

Сменить место — на странице **Setup & Status → Models folder**, или переменной
окружения `ASSETFURNACE_MODELS=<путь>`. Личный `HF_HOME` / `U2NET_HOME` пользователя
всегда имеет приоритет.

## 6. Запуск

| Команда | Что делает |
|---|---|
| `run.bat` (Windows) / `./lab` (Linux) | студия в браузере |
| `run_mcp.bat` / `.\lab mcp` | MCP-сервер для агентов (Claude Code, Codex, …): `claude mcp add unimate -- "<папка>\lab.cmd" mcp` |
| `run_inference.bat` | демо UniMate: три примера (собака, орёл, акула) → GLB/FBX с анимацией |

## 7. Частые проблемы

| Симптом | Что делать |
|---|---|
| `Python 3.10 was not found` | `winget install Python.Python.3.10` (или с python.org), перезапустить `install.bat` |
| `uv installed, but this window cannot see it yet` | открыть **новое** окно PowerShell/cmd и запустить установщик ещё раз |
| Драйвер NVIDIA старый | студия поставится, но PyTorch будет CPU-only, а Pixal3D не соберётся — обновите драйвер до 575+ |
| Blender not found | Finish будет недоступен; поставить Blender 4.2+ с blender.org |
| macOS | UniMate (движение) не работает на Mac — нужен Windows/Linux + NVIDIA |
| Тесты на Windows падают с UnicodeDecodeError | `set PYTHONUTF8=1` перед pytest |
| Движок отказывает на Setup | Pixal3D — только Windows+NVIDIA; TRELLIS.2 и Hunyuan3D-2.1 — только Linux+NVIDIA |

## 8. Примечания

- **Веса не качаются молча**: каждая кнопка Setup и `setup.bat` сначала называют
  размер и спрашивают согласие.
- **Первый запуск UniMate** подтягивает текстовый энкодер `flan-t5-base` с Hugging Face
  (нужен интернет).
- **Лицензия**: MIT на всё — см. `LICENSE`.
- Проблемы при установке — `.\lab` выводит ошибку с подсказкой, что именно отсутствует.
