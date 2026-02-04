#include <python3.10/Python.h>

static struct PyModuleDef constants_module = {
  PyModuleDef_HEAD_INIT,
  "constants",
  "math constants",
  -1,
  NULL
};

PyMODINIT_FUNC PyInit_constants(void){
  PyObject *m = PyModule_Create(&constants_module);
  if(!m) return NULL;
  PyModule_AddObject(m, "pi", PyFloat_FromDouble(3.1415926535897932384626433832795));
  PyModule_AddObject(m, "e", PyFloat_FromDouble(2.7182818284590452353602874713527));
  PyModule_AddObject(m, "euler_gamma", PyFloat_FromDouble(0.5772156649015328606065120900824));
  PyModule_AddObject(m, "inf", PyFloat_FromDouble(__builtin_inf()));
  PyModule_AddObject(m, "nan", PyFloat_FromDouble(__builtin_nan("")));
  return m;
}
