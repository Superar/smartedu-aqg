foreach ($method in $(Get-ChildItem .\results)) {
    New-Item ".\evaluation\$($method.BaseName)" -ItemType "directory"
    foreach ($result in $(Get-ChildItem $method.FullName)) {
        python -m smartedu-aqg.evaluation -f $result.FullName `
            -o ".\evaluation\$($method.BaseName)\$($result.BaseName).json"
    }
}
