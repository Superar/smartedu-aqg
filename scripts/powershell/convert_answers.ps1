foreach ($dir in (Get-ChildItem ..\..\Dados\Motivation\Motivation\*)) {
    if (Test-Path -Type Container -LiteralPath $dir.FullName) {
        foreach ($file in Get-ChildItem $dir -Include *.xlsx -Recurse) {
            python .\scripts\utils\xlsx_to_answers.py -f $file.FullName -o "$($dir.Parent.Parent)\all_answers\$($dir.BaseName).json"
        }
    }
}