#include <python3.10/Python.h>
#include <stdbool.h>
#include "../tensor.h"


static void __add_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->length;
  dtype_t dtype = tx->dtype;
  switch(dtype){
    case BOOL: for(size_t i = 0; i < length; i++){ ((bool *)tz->data)[i] = ((bool *)tx->data)[i] | ((bool *)ty->data)[i]; } break;
    case INT8: for(size_t i = 0; i < length; i++){ ((i8 *)tz->data)[i] = ((i8 *)tx->data)[i] + ((i8 *)ty->data)[i]; } break;
    case UINT8: for(size_t i = 0; i < length; i++){ ((u8 *)tz->data)[i] = ((u8 *)tx->data)[i] + ((u8 *)ty->data)[i]; } break;
    case INT16: for(size_t i = 0; i < length; i++){ ((i16 *)tz->data)[i] = ((i16 *)tx->data)[i] + ((i16 *)ty->data)[i]; } break;
    case UINT16: for(size_t i = 0; i < length; i++){ ((u16 *)tz->data)[i] = ((u16 *)tx->data)[i] + ((u16 *)ty->data)[i]; } break;
    case INT32: for(size_t i = 0; i < length; i++){ ((i32 *)tz->data)[i] = ((i32 *)tx->data)[i] + ((i32 *)ty->data)[i]; } break;
    case UINT32: for(size_t i = 0; i < length; i++){ ((u32 *)tz->data)[i] = ((u32 *)tx->data)[i] + ((u32 *)ty->data)[i]; } break;
    case INT64: for(size_t i = 0; i < length; i++){ ((i64 *)tz->data)[i] = ((i64 *)tx->data)[i] + ((i64 *)ty->data)[i]; } break;
    case UINT64: for(size_t i = 0; i < length; i++){ ((u64 *)tz->data)[i] = ((u64 *)tx->data)[i] + ((u64 *)ty->data)[i]; } break;
    case FP32: for(size_t i = 0; i < length; i++){ ((f32 *)tz->data)[i] = ((f32 *)tx->data)[i] + ((f32 *)ty->data)[i]; } break;
    case FP64: for(size_t i = 0; i < length; i++){ ((f64 *)tz->data)[i] = ((f64 *)tx->data)[i] + ((f64 *)ty->data)[i]; } break;
    case FP128: for(size_t i = 0; i < length; i++){ ((f128 *)tz->data)[i] = ((f128 *)tx->data)[i] + ((f128 *)ty->data)[i]; } break;
    case CMPX64: for(size_t i = 0; i < length; i++){ ((c64 *)tz->data)[i] = ((c64 *)tx->data)[i] + ((c64 *)ty->data)[i]; } break;
    case CMPX128: for(size_t i = 0; i < length; i++){ ((c128 *)tz->data)[i] = ((c128 *)tx->data)[i] + ((c128 *)ty->data)[i]; } break;
    case CMPX256: for(size_t i = 0; i < length; i++){ ((c256 *)tz->data)[i] = ((c256 *)tx->data)[i] + ((c256 *)ty->data)[i]; } break;
    default: {
               PyErr_SetString(PyExc_RuntimeError, "Something is wrong I can feel it");
               return;
             }
  }
}

static void __add_tensor_scalar__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->length;
  dtype_t dtype = tx->dtype;
  switch(dtype){
    case BOOL: for(size_t i = 0; i < length; i++){ ((bool *)tz->data)[i] = ((bool *)tx->data)[i] | ((bool *)ty->data)[0]; } break;
    case INT8: for(size_t i = 0; i < length; i++){ ((i8 *)tz->data)[i] = ((i8 *)tx->data)[i] + ((i8 *)ty->data)[0]; } break;
    case UINT8: for(size_t i = 0; i < length; i++){ ((u8 *)tz->data)[i] = ((u8 *)tx->data)[i] + ((u8 *)ty->data)[0]; } break;
    case INT16: for(size_t i = 0; i < length; i++){ ((i16 *)tz->data)[i] = ((i16 *)tx->data)[i] + ((i16 *)ty->data)[0]; } break;
    case UINT16: for(size_t i = 0; i < length; i++){ ((u16 *)tz->data)[i] = ((u16 *)tx->data)[i] + ((u16 *)ty->data)[0]; } break;
    case INT32: for(size_t i = 0; i < length; i++){ ((i32 *)tz->data)[i] = ((i32 *)tx->data)[i] + ((i32 *)ty->data)[0]; } break;
    case UINT32: for(size_t i = 0; i < length; i++){ ((u32 *)tz->data)[i] = ((u32 *)tx->data)[i] + ((u32 *)ty->data)[0]; } break;
    case INT64: for(size_t i = 0; i < length; i++){ ((i64 *)tz->data)[i] = ((i64 *)tx->data)[i] + ((i64 *)ty->data)[0]; } break;
    case UINT64: for(size_t i = 0; i < length; i++){ ((u64 *)tz->data)[i] = ((u64 *)tx->data)[i] + ((u64 *)ty->data)[0]; } break;
    case FP32: for(size_t i = 0; i < length; i++){ ((f32 *)tz->data)[i] = ((f32 *)tx->data)[i] + ((f32 *)ty->data)[0]; } break;
    case FP64: for(size_t i = 0; i < length; i++){ ((f64 *)tz->data)[i] = ((f64 *)tx->data)[i] + ((f64 *)ty->data)[0]; } break;
    case FP128: for(size_t i = 0; i < length; i++){ ((f128 *)tz->data)[i] = ((f128 *)tx->data)[i] + ((f128 *)ty->data)[0]; } break;
    case CMPX64: for(size_t i = 0; i < length; i++){ ((c64 *)tz->data)[i] = ((c64 *)tx->data)[i] + ((c64 *)ty->data)[0]; } break;
    case CMPX128: for(size_t i = 0; i < length; i++){ ((c128 *)tz->data)[i] = ((c128 *)tx->data)[i] + ((c128 *)ty->data)[0]; } break;
    case CMPX256: for(size_t i = 0; i < length; i++){ ((c256 *)tz->data)[i] = ((c256 *)tx->data)[i] + ((c256 *)ty->data)[0]; } break;
    default: {
               PyErr_SetString(PyExc_RuntimeError, "Something is wrong I can feel it");
               return;
             }
  }
}

static void __sub_tensor__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->length;
  dtype_t dtype = tx->dtype;
  switch(dtype){
    case BOOL: for(size_t i = 0; i < length; i++){ ((bool *)tz->data)[i] = ((bool *)tx->data)[i] - ((bool *)ty->data)[i]; } break;
    case INT8: for(size_t i = 0; i < length; i++){ ((i8 *)tz->data)[i] = ((i8 *)tx->data)[i] - ((i8 *)ty->data)[i]; } break;
    case UINT8: for(size_t i = 0; i < length; i++){ ((u8 *)tz->data)[i] = ((u8 *)tx->data)[i] - ((u8 *)ty->data)[i]; } break;
    case INT16: for(size_t i = 0; i < length; i++){ ((i16 *)tz->data)[i] = ((i16 *)tx->data)[i] - ((i16 *)ty->data)[i]; } break;
    case UINT16: for(size_t i = 0; i < length; i++){ ((u16 *)tz->data)[i] = ((u16 *)tx->data)[i] - ((u16 *)ty->data)[i]; } break;
    case INT32: for(size_t i = 0; i < length; i++){ ((i32 *)tz->data)[i] = ((i32 *)tx->data)[i] - ((i32 *)ty->data)[i]; } break;
    case UINT32: for(size_t i = 0; i < length; i++){ ((u32 *)tz->data)[i] = ((u32 *)tx->data)[i] - ((u32 *)ty->data)[i]; } break;
    case INT64: for(size_t i = 0; i < length; i++){ ((i64 *)tz->data)[i] = ((i64 *)tx->data)[i] - ((i64 *)ty->data)[i]; } break;
    case UINT64: for(size_t i = 0; i < length; i++){ ((u64 *)tz->data)[i] = ((u64 *)tx->data)[i] - ((u64 *)ty->data)[i]; } break;
    case FP32: for(size_t i = 0; i < length; i++){ ((f32 *)tz->data)[i] = ((f32 *)tx->data)[i] - ((f32 *)ty->data)[i]; } break;
    case FP64: for(size_t i = 0; i < length; i++){ ((f64 *)tz->data)[i] = ((f64 *)tx->data)[i] - ((f64 *)ty->data)[i]; } break;
    case FP128: for(size_t i = 0; i < length; i++){ ((f128 *)tz->data)[i] = ((f128 *)tx->data)[i] - ((f128 *)ty->data)[i]; } break;
    case CMPX64: for(size_t i = 0; i < length; i++){ ((c64 *)tz->data)[i] = ((c64 *)tx->data)[i] - ((c64 *)ty->data)[i]; } break;
    case CMPX128: for(size_t i = 0; i < length; i++){ ((c128 *)tz->data)[i] = ((c128 *)tx->data)[i] - ((c128 *)ty->data)[i]; } break;
    case CMPX256: for(size_t i = 0; i < length; i++){ ((c256 *)tz->data)[i] = ((c256 *)tx->data)[i] - ((c256 *)ty->data)[i]; } break;
    default: {
               PyErr_SetString(PyExc_RuntimeError, "Something is wrong I can feel it");
               return;
             }
  }
}

static void __sub_tensor_scalar__(const tensor_t *tx, const tensor_t *ty, tensor_t *tz){
  size_t length = tx->length;
  dtype_t dtype = tx->dtype;
  switch(dtype){
    case BOOL: for(size_t i = 0; i < length; i++){ ((bool *)tz->data)[i] = ((bool *)tx->data)[i] - ((bool *)ty->data)[0]; } break;
    case INT8: for(size_t i = 0; i < length; i++){ ((i8 *)tz->data)[i] = ((i8 *)tx->data)[i] - ((i8 *)ty->data)[0]; } break;
    case UINT8: for(size_t i = 0; i < length; i++){ ((u8 *)tz->data)[i] = ((u8 *)tx->data)[i] - ((u8 *)ty->data)[0]; } break;
    case INT16: for(size_t i = 0; i < length; i++){ ((i16 *)tz->data)[i] = ((i16 *)tx->data)[i] - ((i16 *)ty->data)[0]; } break;
    case UINT16: for(size_t i = 0; i < length; i++){ ((u16 *)tz->data)[i] = ((u16 *)tx->data)[i] - ((u16 *)ty->data)[0]; } break;
    case INT32: for(size_t i = 0; i < length; i++){ ((i32 *)tz->data)[i] = ((i32 *)tx->data)[i] - ((i32 *)ty->data)[0]; } break;
    case UINT32: for(size_t i = 0; i < length; i++){ ((u32 *)tz->data)[i] = ((u32 *)tx->data)[i] - ((u32 *)ty->data)[0]; } break;
    case INT64: for(size_t i = 0; i < length; i++){ ((i64 *)tz->data)[i] = ((i64 *)tx->data)[i] - ((i64 *)ty->data)[0]; } break;
    case UINT64: for(size_t i = 0; i < length; i++){ ((u64 *)tz->data)[i] = ((u64 *)tx->data)[i] - ((u64 *)ty->data)[0]; } break;
    case FP32: for(size_t i = 0; i < length; i++){ ((f32 *)tz->data)[i] = ((f32 *)tx->data)[i] - ((f32 *)ty->data)[0]; } break;
    case FP64: for(size_t i = 0; i < length; i++){ ((f64 *)tz->data)[i] = ((f64 *)tx->data)[i] - ((f64 *)ty->data)[0]; } break;
    case FP128: for(size_t i = 0; i < length; i++){ ((f128 *)tz->data)[i] = ((f128 *)tx->data)[i] - ((f128 *)ty->data)[0]; } break;
    case CMPX64: for(size_t i = 0; i < length; i++){ ((c64 *)tz->data)[i] = ((c64 *)tx->data)[i] - ((c64 *)ty->data)[0]; } break;
    case CMPX128: for(size_t i = 0; i < length; i++){ ((c128 *)tz->data)[i] = ((c128 *)tx->data)[i] - ((c128 *)ty->data)[0]; } break;
    case CMPX256: for(size_t i = 0; i < length; i++){ ((c256 *)tz->data)[i] = ((c256 *)tx->data)[i] - ((c256 *)ty->data)[0]; } break;
    default: {
               PyErr_SetString(PyExc_RuntimeError, "Something is wrong I can feel it");
               return;
             }
  }
}

void capsule_destructor(PyObject *capsule){
  tensor_t *t = PyCapsule_GetPointer(capsule, "tensor_t on CPU");
  if(t){
    destroy(t);
    if(t->data){
      free(t->data);
    }
  }
}

static PyObject *add(PyObject *self, PyObject *args){
  PyObject *x, *y;
  if(!PyArg_ParseTuple(args, "OO", &x, &y)) return NULL;
  if(!PyCapsule_CheckExact(x) || !PyCapsule_CheckExact(y)){
    PyErr_SetString(PyExc_RuntimeError, "Both operands must be tensor capsules");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  tensor_t *ty = PyCapsule_GetPointer(y, "tensor_t on CPU");
  if(!tx || !ty){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  tensor_t *tensor = NULL;
  tensor_t *scalar = NULL;
  if(tx->length == ty->length){
    tensor = tx;
    scalar = NULL;
  }else if(tx->length == 1){
    tensor = ty;
    scalar = tx;
  }else if(ty->length == 1){
    tensor = tx;
    scalar = ty;
  }else{
    PyErr_SetString(PyExc_RuntimeError, "Tensor shape mismatch");
    return NULL;
  }
  if(tx->dtype != ty->dtype){
    PyErr_SetString(PyExc_RuntimeError, "Tensor dtype mismatch");
    return NULL;
  }
  tensor_t *out = malloc(sizeof(tensor_t));
  if(!out){
    PyErr_NoMemory();
    return NULL;
  }
  *out = create(tensor->ndim, tensor->shape, CPU, 0, tensor->dtype);
  if(scalar) __add_tensor_scalar__(tensor, scalar, out);
  else __add_tensor__(tx, ty, out);
  return PyCapsule_New(out, "tensor_t on CPU", capsule_destructor);
}

static PyObject *sub(PyObject *self, PyObject *args){
  PyObject *x, *y;
  if(!PyArg_ParseTuple(args, "OO", &x, &y)) return NULL;
  if(!PyCapsule_CheckExact(x) || !PyCapsule_CheckExact(y)){
    PyErr_SetString(PyExc_RuntimeError, "Both operands must be tensor capsules");
    return NULL;
  }
  tensor_t *tx = PyCapsule_GetPointer(x, "tensor_t on CPU");
  tensor_t *ty = PyCapsule_GetPointer(y, "tensor_t on CPU");
  if(!tx || !ty){
    PyErr_SetString(PyExc_RuntimeError, "Invalid tensor capsule");
    return NULL;
  }
  tensor_t *tensor = NULL;
  tensor_t *scalar = NULL;
  if(tx->length == ty->length){
    tensor = tx;
    scalar = NULL;
  }else if(tx->length == 1){
    tensor = ty;
    scalar = tx;
  }else if(ty->length == 1){
    tensor = tx;
    scalar = ty;
  }else{
    PyErr_SetString(PyExc_RuntimeError, "Tensor shape mismatch");
    return NULL;
  }
  if(tx->dtype != ty->dtype){
    PyErr_SetString(PyExc_RuntimeError, "Tensor dtype mismatch");
    return NULL;
  }
  tensor_t *out = malloc(sizeof(tensor_t));
  if(!out){
    PyErr_NoMemory();
    return NULL;
  }
  *out = create(tensor->ndim, tensor->shape, CPU, 0, tensor->dtype);
  if(scalar) __sub_tensor_scalar__(tensor, scalar, out);
  else __sub_tensor__(tx, ty, out);
  return PyCapsule_New(out, "tensor_t on CPU", capsule_destructor);
}

static PyMethodDef methods[] = {
  {"add", add, METH_VARARGS, "add 2 tensors"},
  {"sub", sub, METH_VARARGS, "sub 2 tensors"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "cpu_ops",
  NULL,
  -1,
  methods
};

PyMODINIT_FUNC PyInit_cpu_ops(void){
  return PyModule_Create(&module);
}
