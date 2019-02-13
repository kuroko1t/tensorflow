package main

import (
	"fmt"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	opt := &tf.ContextOptions{Async: false}
	ctx, _ := tf.NewContext(opt)
	addop := tf.NewOp(ctx, "Add")
	a := []float32{1.1, 1.2, 1.3}
	b := []float32{1.1, 1.2, 1.3}
	ats, _ := tf.NewTensor(a)
	bts, _ := tf.NewTensor(a)
	ahandle, _ := tf.NewTensorHandle(ats)
	bhandle, _ := tf.NewTensorHandle(bts)

	tf.AddInput(addop, ahandle)
	tf.AddInput(addop, bhandle)
	datatype :=tf.TensorHandleDataType(ahandle)
	tf.SetAttrType(addop, "T", datatype)
	handle := &tf.TensorHandle{}
	num_retvals := 1
	tf.Execute(addop, handle, num_retvals)
	fmt.Println("a = ",a)
	fmt.Println("b = ",b)
	fmt.Println("a+b = ",tf.HandleToGo(handle))
}
