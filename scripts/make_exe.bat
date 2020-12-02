pyinstaller ^
    --noconfirm ^
    --log-level=WARN ^
    --clean ^
    --onedir ^
    --console ^
    --paths="niclassify;niclassify\core" ^
    --add-data="niclassify\core\utilities\config;config" ^
    --add-data="niclassify\tkgui\dialogs\messages;dialogs" ^
    --add-data="niclassify\core\scripts;niclassify\core\scripts" ^
    --add-data="niclassifyenv\Lib\site-packages\Bio\Align\substitution_matrices\data;Bio\Align\substitution_matrices\data" ^
    --add-data="niclassify\bin;niclassify\bin" ^
"./niclassify/gui.py"
