$texts_path = "..\..\Dados\Motivation\all_texts"
$answers_path = "..\..\Dados\Motivation\all_answers"

foreach ($file in $(Get-ChildItem $texts_path)) {
    $text_title = $file.BaseName.Split('-')[0]
    $answers_file = "${answers_path}\${text_title}.json"
    $out_file = ".\results\sim_bert\${text_title}.json"
    if (Test-Path $answers_file) {
        python -m smartedu-aqg.methods.bert -f $file -a $answers_file `
                                            -s bert -o $out_file
    }
}