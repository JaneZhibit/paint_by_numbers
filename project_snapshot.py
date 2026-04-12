#!/usr/bin/env python3
"""
Генератор снимков проекта с отслеживанием изменений
Создает версионированные снимки проекта и отчеты об изменениях
"""

import os
import re
import hashlib
import difflib
from pathlib import Path
from datetime import datetime

EXCLUDE_DIRS = {
    '__pycache__', '.git', '.vscode', 'node_modules', '.pytest_cache',
    'venv', 'env', '.env', 'migrations', '.venv', '.idea', '.project_snapshots', 'theory'
}

EXCLUDE_FILES = {
    '.gitignore', '.dockerignore', '.DS_Store', 'Thumbs.db',
    '*.pyc', '*.pyo', '*.pyd', '.env', '*.log', 'project_snapshot.py'
}

INCLUDE_EXTENSIONS = {
    '.py', '.html', '.css', '.js', '.yml', '.yaml', '.txt', '.md',
    '.sql', '.conf', '.sh', '.json', '.xml'
}

def should_include_file(file_path):
    """Проверить, нужно ли включать файл в документацию"""
    file_name = os.path.basename(file_path)

    for exclude in EXCLUDE_FILES:
        if exclude.startswith('*'):
            if file_name.endswith(exclude[1:]):
                return False
        elif file_name == exclude:
            return False

    ext = os.path.splitext(file_name)[1].lower()
    return ext in INCLUDE_EXTENSIONS

def should_include_dir(dir_name):
    """Проверить, нужно ли включать директорию в документацию"""
    return dir_name not in EXCLUDE_DIRS

def get_project_structure(root_path):
    """Получить структуру проекта"""
    structure = []
    root_path = Path(root_path)

    for item in sorted(root_path.rglob('*')):
        if item.is_file():
            parent_dirs = item.relative_to(root_path).parts[:-1]
            should_exclude = any(parent_dir in EXCLUDE_DIRS for parent_dir in parent_dirs)

            if not should_exclude and should_include_file(item):
                relative_path = item.relative_to(root_path)
                structure.append(relative_path)

    return structure

def read_file_content(file_path, max_lines=500):
    """Прочитать содержимое файла с ограничением по строкам"""
    encodings = ['utf-8-sig', 'utf-8', 'utf-16-le', 'utf-16-be', 'cp1251', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
                if len(lines) > max_lines:
                    lines = lines[:max_lines] + [f"\n... (пропущено {len(lines) - max_lines} строк)\n"]
                return ''.join(lines)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return f"[Ошибка чтения файла: {e}]"
    
    return f"[Ошибка: Не удалось декодировать файл ни в одной кодировке]"

def extract_files_from_docs(doc_file):
    """Извлечь содержимое файлов из документации"""
    files_content = {}

    try:
        with open(doc_file, 'r', encoding='utf-8') as f:
            content = f.read()

        pattern = r'### (.*?)\n```\n(.*?)\n```'
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            file_path = match.group(1).strip()
            file_content = match.group(2)
            files_content[file_path] = file_content

        return files_content
    except Exception as e:
        print(f"❌ Ошибка при чтении документации: {e}")
        return {}

def get_file_hash(content):
    """Получить хеш содержимого файла"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def compare_files(current_content, doc_content):
    """Сравнить содержимое файла с версией в документации"""
    current_hash = get_file_hash(current_content)
    doc_hash = get_file_hash(doc_content)
    return current_hash != doc_hash

def get_diff_with_context(current_content, doc_content, context_lines=3):
    """Получить diff с контекстом вокруг изменений"""
    current_lines = current_content.splitlines(keepends=True)
    doc_lines = doc_content.splitlines(keepends=True)

    diff = list(difflib.unified_diff(
        doc_lines,
        current_lines,
        lineterm='',
        n=context_lines
    ))

    if not diff:
        return None

    filtered_diff = []
    for line in diff[3:]:
        if line.startswith(' ') or line.startswith('+'):
            filtered_diff.append(line)

    return ''.join(filtered_diff) if filtered_diff else None

def generate_tree_structure(root_path, structure):
    """Сгенерировать текстовое представление дерева структуры"""
    tree_lines = []
    path_dict = {}

    for file_path in structure:
        parts = file_path.parts
        current = path_dict
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = None

    def build_tree(d, prefix='', is_last=True):
        items = list(d.items())
        for i, (name, subdict) in enumerate(items):
            is_last_item = i == len(items) - 1
            current_prefix = '└── ' if is_last_item else '├── '
            tree_lines.append(f"{prefix}{current_prefix}{name}")

            if subdict is not None:
                next_prefix = prefix + ('    ' if is_last_item else '│   ')
                build_tree(subdict, next_prefix)

    tree_lines.append(root_path.name + '/')
    build_tree(path_dict)
    return '\n'.join(tree_lines)

def get_latest_snapshot_version(snapshots_dir):
    """Получить номер последней версии"""
    if not snapshots_dir.exists():
        return 0

    versions = []
    for item in snapshots_dir.iterdir():
        if item.is_dir() and item.name.startswith('v'):
            try:
                version_num = int(item.name[1:])
                versions.append(version_num)
            except ValueError:
                pass

    return max(versions) if versions else 0

def create_snapshot():
    """Основная функция создания снимка проекта"""
    project_root = Path(__file__).parent
    snapshots_dir = project_root / '.project_snapshots'
    snapshots_dir.mkdir(exist_ok=True)

    # Получить номер версии
    current_version = get_latest_snapshot_version(snapshots_dir) + 1
    version_dir = snapshots_dir / f'v{current_version}'
    version_dir.mkdir(exist_ok=True)

    snapshot_file = version_dir / 'project_snapshot.txt'
    diff_file = version_dir / 'changes.txt'
    metadata_file = version_dir / 'metadata.txt'

    print(f"🔍 Анализ проекта: {project_root}")
    print(f"📦 Создание снимка версии {current_version}")

    # Получить структуру проекта
    structure = get_project_structure(project_root)
    print(f"📁 Найдено файлов: {len(structure)}")

    # Создать снимок проекта
    doc_content = []
    doc_content.append(f"# Снимок проекта версия {current_version}")
    doc_content.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc_content.append("")
    doc_content.append("## Структура проекта")
    doc_content.append("```\n" + generate_tree_structure(project_root, structure) + "\n```")
    doc_content.append("\n## Содержимое файлов")

    for file_path in sorted(structure):
        full_path = project_root / file_path
        relative_path = str(file_path).replace('\\', '/')

        doc_content.append(f"\n### {relative_path}")
        doc_content.append("```")
        content = read_file_content(full_path)
        doc_content.append(content)
        doc_content.append("```")

    # Записать снимок
    try:
        with open(snapshot_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(doc_content))
        print(f"✅ Снимок создан: {snapshot_file}")
    except Exception as e:
        print(f"❌ Ошибка при создании снимка: {e}")
        return

    # Если есть предыдущая версия, создать отчет об изменениях
    if current_version > 1:
        prev_version_dir = snapshots_dir / f'v{current_version - 1}'
        prev_snapshot_file = prev_version_dir / 'project_snapshot.txt'

        if prev_snapshot_file.exists():
            print(f"📊 Сравнение с версией {current_version - 1}")

            # Извлечь содержимое файлов из предыдущего снимка
            prev_files = extract_files_from_docs(prev_snapshot_file)

            # Найти измененные файлы
            changed_files = []
            new_files = []
            deleted_files = []

            for file_path in structure:
                relative_path_str = str(file_path).replace('\\', '/')
                full_path = project_root / file_path
                current_content = read_file_content(full_path)

                if relative_path_str in prev_files:
                    prev_content = prev_files[relative_path_str]
                    if compare_files(current_content, prev_content):
                        changed_files.append((file_path, current_content, prev_content))
                else:
                    new_files.append((file_path, current_content))

            # Найти удаленные файлы
            for prev_file_path in prev_files.keys():
                if not any(str(f).replace('\\', '/') == prev_file_path for f in structure):
                    deleted_files.append(prev_file_path)

            # Создать отчет об изменениях
            changes_content = []
            changes_content.append(f"# Отчет об изменениях версия {current_version}")
            changes_content.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            changes_content.append(f"Сравнение: v{current_version - 1} → v{current_version}")
            changes_content.append("")

            # Статистика
            changes_content.append("## Статистика")
            changes_content.append(f"- Измененные файлы: {len(changed_files)}")
            changes_content.append(f"- Новые файлы: {len(new_files)}")
            changes_content.append(f"- Удаленные файлы: {len(deleted_files)}")
            changes_content.append("")

            # Структура проекта (только измененные файлы)
            if changed_files or new_files:
                changed_paths = [f[0] for f in changed_files] + [f[0] for f in new_files]
                changes_content.append("## Структура проекта (только измененные и новые файлы)")
                changes_content.append("```")
                changes_content.append(generate_tree_structure(project_root, changed_paths))
                changes_content.append("```")
                changes_content.append("")

            # Измененные файлы
            if changed_files:
                changes_content.append("## Измененные файлы")
                for file_path, current_content, prev_content in sorted(changed_files, key=lambda x: str(x[0])):
                    relative_path = str(file_path).replace('\\', '/')
                    diff = get_diff_with_context(current_content, prev_content, context_lines=3)

                    if diff:
                        changes_content.append(f"\n### {relative_path}")
                        changes_content.append("```diff")
                        changes_content.append(diff)
                        changes_content.append("```")

            # Новые файлы
            if new_files:
                changes_content.append("\n## Новые файлы")
                for file_path, content in sorted(new_files, key=lambda x: str(x[0])):
                    relative_path = str(file_path).replace('\\', '/')
                    changes_content.append(f"\n### {relative_path}")
                    changes_content.append("```")
                    changes_content.append(content)
                    changes_content.append("```")

            # Удаленные файлы
            if deleted_files:
                changes_content.append("\n## Удаленные файлы")
                for file_path in sorted(deleted_files):
                    changes_content.append(f"- {file_path}")

            # Записать отчет об изменениях
            try:
                with open(diff_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(changes_content))
                print(f"✅ Отчет об изменениях создан: {diff_file}")
            except Exception as e:
                print(f"❌ Ошибка при создании отчета: {e}")

    # Создать файл метаданных
    metadata_content = []
    metadata_content.append(f"version: {current_version}")
    metadata_content.append(f"timestamp: {datetime.now().isoformat()}")
    metadata_content.append(f"files_count: {len(structure)}")

    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(metadata_content))
    except Exception as e:
        print(f"❌ Ошибка при создании метаданных: {e}")

    print("")
    print(f"📈 Итого:")
    print(f"   Версия: {current_version}")
    print(f"   Файлов в снимке: {len(structure)}")
    print(f"   Директория: {version_dir}")

if __name__ == "__main__":
    create_snapshot()
