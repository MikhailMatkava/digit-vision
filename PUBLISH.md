# Как опубликовать на GitHub

Основной репозиторий: https://github.com/MikhailMatkava/digit-vision.
Ниже — инструкции для публикации собственной копии проекта.
Никому не отправляй пароль или токен в чат.

## Через сайт (без команд Git)

1. Распакуй архив digit-vision.zip.
2. На https://github.com/new создай репозиторий `digit-vision`, выбери Public,
   чтобы рекрутер мог открыть ссылку. Не добавляй автоматический README.
3. Открой «uploading an existing file» и загрузи содержимое папки digit-vision,
   сохраняя вложенные папки. Не загружай .git, .venv, __pycache__ и artifacts.
4. Сделай commit. При загрузке через сайт скрытые .gitignore и .github могут быть
   пропущены системой: добавь их отдельно либо используй вариант ниже.
5. Проверь README, изображения reports и вкладку Actions, если workflow загружен.

## Через Git и GitHub CLI

Архив включает исходники, но не служебную папку .git. Выполни из распакованной папки:

```bash
git init -b main
git config user.name "YOUR_NAME"
git config user.email "YOUR_GITHUB_NOREPLY_EMAIL"
git add .
git commit -m "Add reproducible digit classification baseline"
gh auth login
gh repo create digit-vision --public --source=. --remote=origin --push
```

Замени YOUR_NAME и YOUR_GITHUB_NOREPLY_EMAIL своими значениями; noreply-адрес доступен
в настройках email GitHub. Команда --public делает код открытым всем.
Если имя digit-vision уже занято в твоём аккаунте, выбери другое имя.
Для второго варианта требуется установленный GitHub CLI: https://cli.github.com/.

Для формы заявки используй URL созданного репозитория из вывода gh или браузера.
Перед отправкой запусти проект у себя и разберись с STUDY_GUIDE.md.
