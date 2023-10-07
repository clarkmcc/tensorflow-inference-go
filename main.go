package main

import (
	"context"
	_ "embed"
	"encoding/json"
	signature "example.com/example/signature/host"
	"fmt"
	"github.com/loopholelabs/scale"
	"github.com/loopholelabs/scale/scalefunc"
	"time"
)

//go:embed local-model-latest.scale
var model []byte

//go:embed examples/example0.json
var example0 []byte

//go:embed examples/example1.json
var example1 []byte

//go:embed examples/example2.json
var example2 []byte

//go:embed examples/example3.json
var example3 []byte

//go:embed examples/example4.json
var example4 []byte

//go:embed examples/example5.json
var example5 []byte

//go:embed examples/example6.json
var example6 []byte

//go:embed examples/example7.json
var example7 []byte

//go:embed examples/example8.json
var example8 []byte

//go:embed examples/example9.json
var example9 []byte

func main() {
	sf := new(scalefunc.Schema)
	err := sf.Decode(model)
	if err != nil {
		panic(err)
	}
	s, err := scale.New(scale.NewConfig(signature.New).WithFunction(sf))
	if err != nil {
		panic(err)
	}
	instance, err := s.Instance()
	if err != nil {
		panic(err)
	}

	for i, example := range loadExamples() {
		sig := signature.New()
		sig.Context.Pixels = example
		start := time.Now()
		err = instance.Run(context.Background(), sig)
		if err != nil {
			panic(err)
		}
		if sig.Context.Digit != uint32(i) {
			panic(fmt.Sprintf("expected %d, got %d", i, sig.Context.Digit))
		} else {
			fmt.Printf("Successfully classified the digit %d in %s\n", sig.Context.Digit, time.Since(start))
		}
	}
}

func loadExamples() [][]uint32 {
	var examples [][]uint32
	for _, example := range [][]byte{example0, example1, example2, example3, example4, example5, example6, example7, example8, example9} {
		var pixels []uint32
		err := json.Unmarshal(example, &pixels)
		if err != nil {
			panic(err)
		}
		examples = append(examples, pixels)
	}
	return examples
}
