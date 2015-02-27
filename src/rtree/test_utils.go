package rtree

import "strconv"
import "fmt"

func stofval(s string) *Value {
	float, _ := strconv.ParseFloat(s, 64)
	return &Value{float}
}

func printStringList(lst *[]string) {
	fmt.Printf(strings.Join(*lst, ","))
}
