# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['niclassify\\gui.py'],
             pathex=['niclassify', 'niclassify\\core', 'C:\\Users\\J C\\Documents\\GitHub\\niclassify'],
             binaries=[],
             datas=[('niclassify\\core\\utilities\\config', 'config'), ('niclassify\\tkgui\\dialogs\\messages', 'dialogs'), ('niclassify\\core\\scripts', 'niclassify\\core\\scripts'), ('niclassifyenv\\Lib\\site-packages\\Bio\\Align\\substitution_matrices\\data', 'Bio\\Align\\substitution_matrices\\data'), ('niclassify\\bin', 'niclassify\\bin')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='gui',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
