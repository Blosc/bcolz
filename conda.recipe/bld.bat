xcopy /e "%RECIPE_DIR%\.." "%SRC_DIR%"
SET BLD_DIR=%CD%
cd /D "%RECIPE_DIR%\.."
FOR /F "delims=" %%i IN ('git describe --tags') DO set BCOLZ_VERSION=%%i
echo.%BCOLZ_VERSION% | %PYTHON% .\conda.recipe\version.py > %SRC_DIR%\__conda_version__.txt
%PYTHON% setup.py --quiet install
