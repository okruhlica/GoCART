package decision_tree

import "strconv"
import "fmt"
import "strings"

func stofval(s string) interface{} {
	float, _ := strconv.ParseFloat(s, 64)
	return float
}

func printStringList(lst *[]string) {
	fmt.Printf(strings.Join(*lst, ","))
}
