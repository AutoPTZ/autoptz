; Inno Setup script for AutoPTZ (Windows x64 installer).
;
; Build the onedir bundle first (packaging\build_windows.ps1), then compile:
;   iscc /DMyAppVersion=2.0.0 packaging\autoptz.iss
; Produces dist\AutoPTZ-<version>-windows-x64-setup.exe with Start-menu /
; optional desktop shortcuts and an uninstaller.

#define MyAppName "AutoPTZ"
#ifndef MyAppVersion
  #define MyAppVersion "2.0.0"
#endif
#define MyAppPublisher "AutoPTZ"
#define MyAppURL "https://github.com/AutoPTZ/autoptz"
#define MyAppExeName "AutoPTZ.exe"

[Setup]
; Stable AppId so upgrades replace the prior install (do not change between releases).
AppId={{B7F4B2E2-6F4A-4E0B-9C3E-AUTOPTZ200000}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
OutputDir=..\dist
OutputBaseFilename=AutoPTZ-{#MyAppVersion}-windows-x64-setup
SetupIconFile=AutoPTZ.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
; In-app updates run this installer silently.  Let Setup close the running
; AutoPTZ so its files can be replaced, but don't let the Restart Manager
; relaunch it — the silent [Run] entry below does the relaunch deterministically
; (avoids a double launch when an older app build also passed /RESTARTAPPLICATIONS).
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\AutoPTZ\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Interactive installs: the usual "launch AutoPTZ" checkbox on the Finished page.
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent
; Silent installs (the in-app auto-updater): relaunch AutoPTZ automatically, since
; the Finished-page checkbox above is skipped when silent.  runasoriginaluser so
; the relaunched app runs as the user, not the elevated installer.
Filename: "{app}\{#MyAppExeName}"; Flags: nowait runasoriginaluser; Check: IsSilentInstall

[Code]
function IsSilentInstall(): Boolean;
begin
  { True for both /SILENT and /VERYSILENT — i.e. an in-app auto-update. }
  Result := WizardSilent();
end;
