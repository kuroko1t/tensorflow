package tensorflow

// #include <stdlib.h>
// #include "tensorflow/c/c_api.h"
// #include "tensorflow/c/eager/c_api.h"
import "C"
import (
	"log"
)
type Tape struct {
	c *C.TensorTape
}

type EOp struct {
   c *C.TFE_Op
}

func NewOp(ctx *Context, opname string) (*EOp) {
	status := newStatus()
	op := C.TFE_NewOp(ctx.c, C.CString(opname), status.c)
	eop := &EOp{c: op}
	if err := status.Err(); err != nil {
		log.Fatal(err)
	}
	status.finalizer()
	return eop
}

func AddInput(op *EOp, a *TensorHandle) {
	status := newStatus()
	C.TFE_OpAddInput(op.c, a.c, status.c);
	if err := status.Err(); err != nil {
		log.Fatal(err)
	}
	status.finalizer()
}

func SetAttrType(op *EOp, attrname string, tp DataType) {
	C.TFE_OpSetAttrType(op.c, C.CString(attrname), C.TF_DataType(tp))
}

func TensorHandleDataType(a *TensorHandle) DataType {
	return DataType(C.TFE_TensorHandleDataType(a.c))
}

func Execute(op *EOp, retvals *TensorHandle, num_retvals int) {
	status := newStatus()
	num := C.int(num_retvals)
	C.TFE_Execute(op.c, &retvals.c, &num, status.c)
	if err := status.Err(); err != nil {
		log.Fatal(err)
	}
	status.finalizer()
}

func HandleToGo(th *TensorHandle) interface{} {
	t , err := th.ToTensor()
	if err != nil {
		log.Fatal(err)
	}
	return t.Value()
}
