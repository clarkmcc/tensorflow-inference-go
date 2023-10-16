// Code generated by scale-signature v0.4.3, DO NOT EDIT.
// output: signature

package signature

import (
	"errors"
	"github.com/loopholelabs/polyglot"
)

var (
	NilDecode   = errors.New("cannot decode into a nil root struct")
	InvalidEnum = errors.New("invalid enum value")
)

type Context struct {
	Digit uint32

	Pixels []uint32
}

func NewContext() *Context {
	return &Context{

		Digit: 0,

		Pixels: make([]uint32, 0, 784),
	}
}

func (x *Context) Encode(b *polyglot.Buffer) {
	e := polyglot.Encoder(b)
	if x == nil {
		e.Nil()
	} else {

		e.Uint32(x.Digit)

		e.Slice(uint32(len(x.Pixels)), polyglot.Uint32Kind)
		for _, a := range x.Pixels {
			e.Uint32(a)
		}

	}
}

func DecodeContext(x *Context, b []byte) (*Context, error) {
	d := polyglot.GetDecoder(b)
	defer d.Return()
	return _decodeContext(x, d)
}

func _decodeContext(x *Context, d *polyglot.Decoder) (*Context, error) {
	if d.Nil() {
		return nil, nil
	}

	err, _ := d.Error()
	if err != nil {
		return nil, err
	}

	if x == nil {
		x = NewContext()
	}

	x.Digit, err = d.Uint32()
	if err != nil {
		return nil, err
	}

	sliceSizePixels, err := d.Slice(polyglot.Uint32Kind)
	if err != nil {
		return nil, err
	}

	if uint32(len(x.Pixels)) != sliceSizePixels {
		x.Pixels = make([]uint32, sliceSizePixels)
	}

	for i := uint32(0); i < sliceSizePixels; i++ {
		x.Pixels[i], err = d.Uint32()
		if err != nil {
			return nil, err
		}
	}

	return x, nil
}
