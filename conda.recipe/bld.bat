xcopy /e "%RECIPE_DIR%\.." "%SRC_DIR%"
SET BLD_DIR=%CD%
cd /D "%RECIPE_DIR%\.."
copy "%SRC_DIR%\VERSION" "%SRC_DIR%\__conda_version__.txt"
%PYTHON% setup.py --quiet install
