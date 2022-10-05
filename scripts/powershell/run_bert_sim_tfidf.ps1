$texts_path = "..\..\Dados\Motivation\all_texts"
$answers_path = "..\..\Dados\Motivation\all_answers"
$idf_path = "..\..\Dados\CETEMPublico\CETEMPublico_mini"

foreach ($file in $(Get-ChildItem $texts_path)) {
    $text_title = $file.BaseName.Split('-')[0]
    $answers_file = "${answers_path}\${text_title}.json"
    $out_file = ".\results\sim_tfidf\${text_title}.json"
    if (Test-Path $answers_file) {
        python -m smartedu-aqg.methods.bert -f $file -a $answers_file `
                                            -s tfidf -o $out_file `
                                            -t $idf_path
    }
}