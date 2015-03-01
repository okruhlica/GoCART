# GoCART
Decision tree classifier implementation in Go language.

== Input == 
float64 predictor values, categorical (0/1) target values

== Output == 
Classifier.

Usage:

import 'rtree'

...
observations := loadSomeObservations()
t := new(rtree)
t.InitRoot(getSettings("supergrow"), *observations).Expand(true)
...
t.Classify(map[string]*Value{ 
	"attr1": &Value{1.0},
	"attr2": &Value{17.0},
})