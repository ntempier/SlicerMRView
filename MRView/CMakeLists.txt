set(MODULE_NAME MRView)

set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  MRViewLib/__init__.py
  MRViewLib/layers.py
  MRViewLib/mrtrix_io.py
  MRViewLib/sh.py
  MRViewLib/tracts.py
  MRViewLib/vtk_utils.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  )

slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )
