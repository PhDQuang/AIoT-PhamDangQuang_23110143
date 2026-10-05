param([string]$ProjectRoot)
$ErrorActionPreference='Stop'
$content=Get-Content (Join-Path $ProjectRoot 'reports/presentation_content.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$out=Join-Path $ProjectRoot 'reports/PhamDangQuang_final.pptx'
$qa=Join-Path $ProjectRoot 'reports/work/ppt_preview'
New-Item -ItemType Directory -Force -Path $qa | Out-Null
function RGB($r,$g,$b){return [int]($r+256*$g+65536*$b)}
$navy=RGB 17 39 60; $teal=RGB 0 128 122; $gray=RGB 72 87 103
$app=New-Object -ComObject PowerPoint.Application
try {
 $deck=$app.Presentations.Add(0)
 $deck.PageSetup.SlideWidth=960; $deck.PageSetup.SlideHeight=540
 function Text($s,$value,$x,$y,$w,$h,$size,$color,$bold=$false){
  $shape=$s.Shapes.AddTextbox(1,$x,$y,$w,$h)
  $shape.TextFrame.MarginLeft=0; $shape.TextFrame.MarginRight=0
  $shape.TextFrame.MarginTop=0; $shape.TextFrame.MarginBottom=0
  $shape.TextFrame.WordWrap=-1
  $range=$shape.TextFrame.TextRange; $range.Text=[string]$value
  $range.Font.Name='Arial'; $range.Font.Size=$size; $range.Font.Color.RGB=$color
  if($bold){$range.Font.Bold=-1}
  $range.ParagraphFormat.SpaceAfter=12
  return $shape
 }
 $index=0
 foreach($item in $content){
  $index++; $slide=$deck.Slides.Add($index,12)
  $slide.FollowMasterBackground=0
  if($item.cover){
   $slide.Background.Fill.ForeColor.RGB=$navy
   $null=Text $slide $item.title 65 130 830 140 42 (RGB 255 255 255) $true
   $null=Text $slide $item.subtitle 65 290 825 80 24 (RGB 119 221 208)
   $null=Text $slide ($item.body -join "`r") 65 390 825 110 18 (RGB 225 236 245)
  } else {
   $slide.Background.Fill.ForeColor.RGB=RGB 255 255 255
   $null=Text $slide $item.title 52 30 856 85 32 $navy $true
   $null=Text $slide ([string]$index) 887 509 25 18 10 $gray
   if($item.image){
    $file=Join-Path $ProjectRoot ('reports/assets/'+$item.image+'.png')
    if(!(Test-Path -LiteralPath $file)){$file=Join-Path $ProjectRoot ('reports/work/figures/'+$item.image+'.png')}
    $pic=$slide.Shapes.AddPicture($file,0,-1,0,0,-1,-1)
    $ratio=$pic.Width/$pic.Height
    $maxW=850; $maxH=325
    if($ratio -gt $maxW/$maxH){$pic.Width=[single]$maxW; $pic.Height=[single]($maxW/$ratio)}else{$pic.Height=[single]$maxH; $pic.Width=[single]($maxH*$ratio)}
    $pic.Left=[single]((960-$pic.Width)/2); $pic.Top=[single](115+(325-$pic.Height)/2)
    $null=Text $slide ($item.body -join "`r") 55 452 850 53 17 $gray
   } elseif($item.table){
    $rows=$item.table.Count; $cols=$item.table[0].Count
    $height=[math]::Min(295,45*$rows)
    $shape=$slide.Shapes.AddTable($rows,$cols,55,125,850,$height)
    $table=$shape.Table
    for($r=1;$r -le $rows;$r++){
     for($c=1;$c -le $cols;$c++){
      $cell=$table.Cell($r,$c).Shape
      $cell.TextFrame.MarginLeft=12; $cell.TextFrame.MarginRight=10
      $cell.TextFrame.MarginTop=8; $cell.TextFrame.MarginBottom=5
      $cell.TextFrame.VerticalAnchor=3
      $tr=$cell.TextFrame.TextRange; $tr.Text=[string]$item.table[$r-1][$c-1]
      $tr.Font.Name='Arial'; $tr.Font.Size=18
      if($r -eq 1){$cell.Fill.ForeColor.RGB=$navy; $tr.Font.Color.RGB=RGB 255 255 255; $tr.Font.Bold=-1}
      else{$cell.Fill.ForeColor.RGB=RGB 240 245 248; $tr.Font.Color.RGB=$navy}
     }
    }
    if($item.body){$null=Text $slide ($item.body -join "`r") 55 (145+$height) 850 110 19 $gray}
   } else {
    $top=135
    foreach($line in $item.body){
     $null=Text $slide $line 60 $top 835 77 23 $gray
     $top+=86
    }
   }
  }
  if($item.note){
   foreach($shape in $slide.NotesPage.Shapes){
    if($shape.Type -eq 14 -and $shape.PlaceholderFormat.Type -eq 2){$shape.TextFrame.TextRange.Text=$item.note}
   }
  }
 }
 $deck.SaveAs($out,24)
 $overflow=@()
 foreach($slide in $deck.Slides){
  $slide.Export((Join-Path $qa ('slide-{0:D2}.png' -f $slide.SlideIndex)),'PNG',1600,900)
  foreach($shape in $slide.Shapes){
   if($shape.HasTextFrame -and $shape.TextFrame.HasText){
    if($shape.TextFrame2.TextRange.BoundHeight -gt ($shape.Height+3)){$overflow+=('slide '+$slide.SlideIndex+': '+$shape.TextFrame.TextRange.Text)}
   }
  }
 }
 $overflow | Set-Content (Join-Path $qa 'overflow.txt') -Encoding UTF8
 Write-Output "PPTX: $($deck.Slides.Count) slides; overflow: $($overflow.Count)"
 $deck.Close()
} finally {$app.Quit()}
