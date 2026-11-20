$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"

Set-Location $Root
& $Python "demo\create_fixtures.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$tasks = @(
    [ordered]@{
        Name = "text-to-bengali-audio"
        Task = "Translate this text to Bengali and synthesize it as female Bengali speech"
        Files = "demo\inputs\hello.txt"
        Output = "out\demo-text-bengali"
    },
    [ordered]@{
        Name = "hindi-srt"
        Task = "Transcribe this Hindi voice note to English and return the result as an English SRT subtitle"
        Files = "demo\inputs\voice-note.wav"
        Output = "out\demo-hindi-srt"
    },
    [ordered]@{
        Name = "worksheet-marathi-markdown"
        Task = "OCR this worksheet photo and translate the text to Marathi Markdown with the table preserved"
        Files = "demo\inputs\worksheet.png"
        Output = "out\demo-worksheet-marathi"
    },
    [ordered]@{
        Name = "telugu-clips-english"
        Task = "Transcribe these five short clips, find the Telugu clips, and translate them to English"
        Files = "demo\inputs\clips"
        Output = "out\demo-telugu-clips"
    },
    [ordered]@{
        Name = "pdf-tamil-summary-audio"
        Task = "OCR this two-page PDF, write an English summary, translate the summary to Tamil, and synthesize Tamil audio"
        Files = "demo\inputs\scanned.pdf"
        Output = "out\demo-pdf-tamil"
    },
    [ordered]@{
        Name = "unsupported-video-french"
        Task = "Translate this video into French"
        Files = "demo\inputs\clip.mp4"
        Output = "out\demo-video-french"
    }
)

$rows = foreach ($item in $tasks) {
    Write-Host ""
    Write-Host "=== $($item.Name) ==="
    & $Python "-m" "bodhan_agent" `
        "--task" $item.Task `
        "--files" $item.Files `
        "--out" $item.Output `
        "--max-cost-inr" "2"
    $exitCode = $LASTEXITCODE

    $reportPath = Join-Path $Root ($item.Output + "\report.json")
    if (Test-Path $reportPath) {
        $report = Get-Content -Raw $reportPath | ConvertFrom-Json
        [PSCustomObject][ordered]@{
            Task = $item.Name
            Status = $report.status
            Cost = ("INR {0:N6}" -f [double]$report.total_cost_inr)
            Time = ("{0:N3}s" -f [double]$report.elapsed_seconds)
            ExitCode = $exitCode
            Output = $item.Output
        }
    } else {
        [PSCustomObject][ordered]@{
            Task = $item.Name
            Status = "failed"
            Cost = "INR 0.000000"
            Time = "0.000s"
            ExitCode = $exitCode
            Output = $item.Output
        }
    }
}

$tableRows = @(
    "| Task | Status | Cost | Time | Output |"
    "|---|---|---:|---:|---|"
)
foreach ($row in $rows) {
    $tableRows += "| $($row.Task) | $($row.Status) | $($row.Cost) | $($row.Time) | ``$($row.Output)`` |"
}
Set-Content -Path (Join-Path $Root "demo\RESULTS.md") -Value (
    "# Demo Results`n`n" + ($tableRows -join "`n") + "`n"
) -Encoding utf8

$rows | Format-Table -AutoSize
Write-Host ""
Write-Host "Results table:"
Write-Host ((($tableRows -join "`n") + "`n"))
