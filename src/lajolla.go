package main

import "fmt"
import "cart"

func main() {
	t := new(cart.CartTree)
	t1 := new(cart.CartTree)
	t2 := new(cart.CartTree)
	t.Good = 100
	t.Bad = 20
	t1.Good = 20
	t1.Bad = 20
	t2.Good = 80
	t2.Bad = 0
	cart.AddLeft(t, t1)
	//	AddRight(t, t2)

	fmt.Printf("Node is leaf? %t", cart.IsLeaf(t))
}
