cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX="%LIBRARY_PREFIX%" -D BUILD_TESTS=OFF
if errorlevel 1 exit 1
cmake --build build
if errorlevel 1 exit 1
cmake --install build
if errorlevel 1 exit 1
