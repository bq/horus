!ifndef VERSION
  !define VERSION 'DEV'
!endif
!addplugindir "nsisPlugins"

; The name of the installer
Name "Horus ${VERSION}"

; The file to write
OutFile "Horus_${VERSION}.exe"

; The default installation directory
InstallDir $PROGRAMFILES\Horus_${VERSION}

; Registry key to check for directory (so if you install again, it will 
; overwrite the old one automatically)
InstallDirRegKey HKLM "Software\Horus_${VERSION}" "Install_Dir"

; Request application privileges for Windows Vista
RequestExecutionLevel admin

; Set the LZMA compressor to reduce size.
SetCompressor /SOLID lzma
;--------------------------------

!include "MUI2.nsh"
!include Library.nsh

!define MUI_ICON "dist/res/horus.ico"
!define MUI_BGCOLOR FFFFFF

; Directory page defines
!define MUI_DIRECTORYPAGE_VERIFYONLEAVE

; Header
#!define MUI_HEADERIMAGE
#!define MUI_HEADERIMAGE_RIGHT
#!define MUI_HEADERIMAGE_BITMAP "header.bmp"
#!define MUI_HEADERIMAGE_BITMAP_NOSTRETCH 
; Don't show the component description box
!define MUI_COMPONENTSPAGE_NODESC

;Do not leave (Un)Installer page automaticly
!define MUI_FINISHPAGE_NOAUTOCLOSE
!define MUI_UNFINISHPAGE_NOAUTOCLOSE

;Run Cura after installing
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_TEXT "Start Horus ${VERSION}"
!define MUI_FINISHPAGE_RUN_FUNCTION "LaunchLink"

; Pages
;!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

; Languages
!insertmacro MUI_LANGUAGE "English"

; Reserve Files
!insertmacro MUI_RESERVEFILE_LANGDLL
ReserveFile '${NSISDIR}\Plugins\InstallOptions.dll'
#ReserveFile "header.bmp"

;--------------------------------

; The stuff to install
Section "Horus ${VERSION}"

  SectionIn RO
  
  ; Set output path to the installation directory.
  SetOutPath $INSTDIR

  ; Put file there
  File /r "dist\"

  ; Remove ini files version
  Delete "$INSTDIR\src\horus\*.ini"
  
  ; Write the installation path into the registry
  WriteRegStr HKLM "SOFTWARE\Horus_${VERSION}" "Install_Dir" "$INSTDIR"
  
  ; Write the uninstall keys for Windows
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Horus_${VERSION}" "DisplayName" "Horus ${VERSION}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Horus_${VERSION}" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Horus_${VERSION}" "NoModify" 1
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Horus_${VERSION}" "NoRepair" 1
  WriteUninstaller "uninstall.exe"

  ; Write start menu entries for all users
  SetShellVarContext all
  
  CreateDirectory "$SMPROGRAMS\Horus ${VERSION}"
  CreateShortCut "$SMPROGRAMS\Horus ${VERSION}\Uninstall Horus ${VERSION}.lnk" "$INSTDIR\uninstall.exe" "" "$INSTDIR\uninstall.exe" 0
  CreateShortCut "$SMPROGRAMS\Horus ${VERSION}\Horus ${VERSION}.lnk" "$INSTDIR\python\python.exe" "$\"$INSTDIR\src\horus.py$\"" "$INSTDIR\res\horus.ico" 0
  
  ; Give all users write permissions in the install directory, so they can read/write profile and preferences files.
  AccessControl::GrantOnFile "$INSTDIR" "(S-1-5-32-545)" "FullAccess"
  
SectionEnd

Function LaunchLink
  ; Write start menu entries for all users
  SetShellVarContext all
  ExecShell "" "$SMPROGRAMS\Horus ${VERSION}\Horus ${VERSION}.lnk"
FunctionEnd

Section "Install FTDI Drivers"
  ; Set output path to the driver directory.
  SetOutPath "$INSTDIR\drivers\"
  File /r "drivers\"
 
  ExecWait '"$INSTDIR\drivers\CDM v2.10.00 WHQL Certified.exe" /lm'
SectionEnd

Section "Install Arduino Drivers"
  ; Set output path to the driver directory.
  SetOutPath "$INSTDIR\drivers\"
  File /r "drivers\"

  ${If} ${RunningX64}
    ExecWait '"$INSTDIR\drivers\dpinst64.exe" /lm'
  ${Else}
    ExecWait '"$INSTDIR\drivers\dpinst32.exe" /lm'
  ${EndIf}
SectionEnd

;Section "Open PLY files with Horus"
;	WriteRegStr HKCR .ply "" "Horus PLY model file"
;	DeleteRegValue HKCR .ply "Content Type"
;	WriteRegStr HKCR "Horus PLY model file\DefaultIcon" "" "$INSTDIR\res\stl.ico,0"
;	WriteRegStr HKCR "Horus PLY model file\shell" "" "open"
;	WriteRegStr HKCR "Horus PLY model file\shell\open\command" "" '"$INSTDIR\python\pythonw.exe" -c "import os; os.chdir(\"$INSTDIR\"); import Horus; Horus.main()" "%1"'
;SectionEnd
;
;Section "Open STL files with Horus"
;	WriteRegStr HKCR .stl "" "Horus STL model file"
;	DeleteRegValue HKCR .stl "Content Type"
;	WriteRegStr HKCR "Horus STL model file\DefaultIcon" "" "$INSTDIR\res\stl.ico,0"
;	WriteRegStr HKCR "Horus STL model file\shell" "" "open"
;	WriteRegStr HKCR "Horus STL model file\shell\open\command" "" '"$INSTDIR\python\pythonw.exe" -c "import os; os.chdir(\"$INSTDIR\"); import Horus; Horus.main()" "%1"'
;SectionEnd

;--------------------------------

; Uninstaller

Section "Uninstall"
  
  ; Remove registry keys
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Horus_${VERSION}"
  DeleteRegKey HKLM "SOFTWARE\Horus_${VERSION}"

  ; Write start menu entries for all users
  SetShellVarContext all

  ; Remove directories used
  RMDir /r "$SMPROGRAMS\Horus ${VERSION}"
  RMDir /r "$INSTDIR"

SectionEnd