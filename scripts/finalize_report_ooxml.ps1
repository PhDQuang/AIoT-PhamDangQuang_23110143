param([string]$ProjectRoot,[string]$ReportName='Bao_cao_PPG_CVAE_AI_cho_IoT')
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.IO.Compression
$path=Join-Path $ProjectRoot "reports/$ReportName.docx"
$archive=[IO.Compression.ZipFile]::Open($path,[IO.Compression.ZipArchiveMode]::Update)
function Read-XmlEntry([string]$Name) {
    $reader=New-Object IO.StreamReader($archive.GetEntry($Name).Open())
    try {[xml]$reader.ReadToEnd()} finally {$reader.Dispose()}
}
function Write-XmlEntry([string]$Name,[xml]$Xml) {
    $archive.GetEntry($Name).Delete()
    $entry=$archive.CreateEntry($Name)
    $stream=$entry.Open()
    $settings=New-Object Xml.XmlWriterSettings; $settings.Encoding=New-Object Text.UTF8Encoding($false)
    $writer=[Xml.XmlWriter]::Create($stream,$settings)
    try {$Xml.Save($writer)} finally {$writer.Dispose();$stream.Dispose()}
}
function MathRun([string]$Text) { '<m:r><m:t xml:space="preserve">'+[Security.SecurityElement]::Escape($Text)+'</m:t></m:r>' }
function Sub([string]$Base,[string]$Index) { '<m:sSub><m:e>'+(MathRun $Base)+'</m:e><m:sub>'+(MathRun $Index)+'</m:sub></m:sSub>' }
function Sup([string]$Base,[string]$Index) { '<m:sSup><m:e>'+$Base+'</m:e><m:sup>'+(MathRun $Index)+'</m:sup></m:sSup>' }
function Frac([string]$Num,[string]$Den) { '<m:f><m:num>'+(MathRun $Num)+'</m:num><m:den>'+(MathRun $Den)+'</m:den></m:f>' }
function Sum([string]$Lower,[string]$Upper,[string]$Body) {
    $hide='';if(-not $Upper){$hide='<m:supHide m:val="1"/>'}
    '<m:nary><m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/>'+$hide+'</m:naryPr><m:sub>'+(MathRun $Lower)+'</m:sub><m:sup>'+(MathRun $Upper)+'</m:sup><m:e>'+$Body+'</m:e></m:nary>'
}
try {
    $xml=Read-XmlEntry 'word/document.xml'
    $ns=New-Object Xml.XmlNamespaceManager($xml.NameTable)
    $ns.AddNamespace('w','http://schemas.openxmlformats.org/wordprocessingml/2006/main')
    $ns.AddNamespace('m','http://schemas.openxmlformats.org/officeDocument/2006/math')
    foreach($node in @($xml.SelectNodes('//w:tl2br|//w:tr2bl',$ns))) {[void]$node.ParentNode.RemoveChild($node)}
    $eq=@()
    $eq += (MathRun 'z = μ + exp(s/2) ⊙ ε,   ε ∼ N(0,I)')
    $eq += (Sub 'L' 'rec')+(MathRun ' = ')+(Frac '1' '512')+(Sum 'i=1' '512' (Sup ((MathRun '(')+(Sub 'x' 'i')+(MathRun ' − ')+(Sub 'x̂' 'i')+(MathRun ')')) '2'))
    $eq += (Sub 'L' 'KL')+(MathRun ' = ')+(Frac '1' '2d')+(Sum 'r=1' 'd' ((MathRun '(')+(Sup (Sub 'μ' 'r') '2')+(MathRun ' + exp(')+(Sub 's' 'r')+(MathRun ') − 1 − ')+(Sub 's' 'r')+(MathRun ')')))+(MathRun ',   d = 16')
    $eq += (Sub 'L' 'HR')+(MathRun ' = |')+(Sub 'g' 'ω')+(MathRun '(x̂) − c|,   c = h/60')
    $eq += (MathRun 'L = ')+(Sub 'L' 'rec')+(MathRun ' + β')+(Sub 'L' 'KL')+(MathRun ' + ')+(Sub 'λ' 'HR')+(Sub 'L' 'HR')
    $eq += (MathRun 'MAE = ')+(Frac '1' '|S|')+(Sum 'j∈S' '' ((MathRun '|')+(Sub 'h̃' 'j')+(MathRun ' − h|')))
    $eq += (MathRun 'V = ')+(Frac '|S|' 'M')
    $eq += (Sub 'A' '3')+(MathRun ' = ')+(Frac '1' 'M')+(Sum 'j∈S' '' ((MathRun 'I(|')+(Sub 'h̃' 'j')+(MathRun ' − h| ≤ 3)')))
    $inside=(Frac '1' '512')+(Sum 'i=1' '512' (Sup ((MathRun '(')+(Sub 'x' 'j,i')+(MathRun ' − ')+(Sub 'x̂' 'j,i')+(MathRun ')')) '2'))
    $eq += (Sub 'RMSE' 'j')+(MathRun ' = ')+'<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>'+$inside+'</m:e></m:rad>'
    $nodes=@($xml.SelectNodes('//m:oMath',$ns))
    if($nodes.Count -ne 9){throw 'Unexpected equation count'}
    for($i=0;$i -lt 9;$i++){
        $fragment=$xml.CreateDocumentFragment()
        $fragment.InnerXml='<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'+$eq[$i]+'</m:oMath>'
        [void]$nodes[$i].ParentNode.ReplaceChild($fragment,$nodes[$i])
    }
    Write-XmlEntry 'word/document.xml' $xml
    $core=Read-XmlEntry 'docProps/core.xml'
    $nsc=New-Object Xml.XmlNamespaceManager($core.NameTable)
    $nsc.AddNamespace('dc','http://purl.org/dc/elements/1.1/')
    $core.SelectSingleNode('//dc:title',$nsc).InnerText='Sinh tín hiệu PPG theo nhịp tim bằng cVAE'
    $core.SelectSingleNode('//dc:creator',$nsc).InnerText='Phạm Đăng Quang'
    Write-XmlEntry 'docProps/core.xml' $core
} finally {$archive.Dispose()}
Write-Output 'Removed diagonal table borders and finalized nine OMML equations'
