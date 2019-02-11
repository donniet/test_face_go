package main

/*
#cgo LDFLAGS: -L../detect_faces/build/armv7l/Release/lib -ldetector -lclassifier -lmulti_modal

#include <stdio.h>

typedef struct detection_t {
  float confidence;
  float label;
  float xmin, xmax, ymin, ymax;
} detection;

typedef struct response_t {
  unsigned long num_detections;
  detection * detections;
} response;

typedef struct detector_t {
  void * face_detector;
} detector;

extern int create_face_detector(
    const char * networkFile,
    const char * networkWeights,
    const char * deviceName,
    detector ** d);

extern response * do_inference(detector * d, void * pix, int stride, int x0, int y0, int x1, int y1);
extern void destroy_response(response * res);
extern void destroy_face_detector(detector * d);

typedef struct classifier_t {
  void * network;
} classifier;

typedef struct classifier_request_t {
  char * data;
  size_t image_width;
  size_t image_height;
} classifier_request;

typedef struct classifier_response_t {
  float * embedding;
  size_t embedding_size;
  float duration;
} classifier_response;

extern classifier * create_classifier(char * networkFile, char * networkWeights, char * deviceName);
extern void destroy_classifier(classifier * c);
extern classifier_response * do_classification(classifier * c, void * data, int stride, int x0, int y0, int x1, int y1);
extern void destroy_classifier_response(classifier_response * r);

float get_embedding_at(float * embedding, int i) {
	return embedding[i];
}

detection * get_response_detection(response * res, unsigned long dex) {
  if(dex < res->num_detections) {
    return res->detections + dex;
  }
  return NULL;
}


typedef struct {
    void * ds;
    size_t dimensions;
} multi_modal_wrapper;

typedef struct {
    float * mean;
    size_t mean_size;
    float standard_deviation;
    size_t sample_count;
} distribution_wrapper;

multi_modal_wrapper * mm_create(size_t dimensions, size_t maximum_nodes);
void mm_destroy(multi_modal_wrapper * wrapper);

void mm_insert(multi_modal_wrapper * wrapper, float * sample, size_t dimensions);
size_t mm_get_count(multi_modal_wrapper * wrapper);
void mm_extract_peaks(multi_modal_wrapper * wrapper, distribution_wrapper ** wrappers, size_t * wrapper_count);
void mm_destroy_peaks(multi_modal_wrapper * wrapper, distribution_wrapper * wrappers, size_t wrapper_count);

distribution_wrapper * get_peak(distribution_wrapper * dist, size_t i) {
	return &dist[i];
}
float get_element(float * arr, size_t i) {
	return arr[i];
}

*/
import "C"

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"io"
	"io/ioutil"
	"log"
	"math"
	"os"
	"time"
	"unsafe"

	"image/jpeg"
)

type MultiModal struct {
	wrapper *C.multi_modal_wrapper
}
type Distribution struct {
	Mean   []float32
	StdDev float32
	Count  int
}

func NewMultiModal(dimensions int, maximumNodes int) MultiModal {
	return MultiModal{
		wrapper: C.mm_create(C.uint(dimensions), C.uint(maximumNodes)),
	}
}
func (mm MultiModal) Close() {
	C.mm_destroy(mm.wrapper)
}
func (mm MultiModal) Insert(vector []float32) {
	dat := make([]C.float, len(vector))
	for i, f := range vector {
		dat[i] = C.float(f)
	}
	C.mm_insert(mm.wrapper, &dat[0], C.uint(len(vector)))
}
func (mm MultiModal) Count() int {
	return int(C.mm_get_count(mm.wrapper))
}
func (mm MultiModal) Peaks() []Distribution {
	var dist *C.distribution_wrapper
	var count C.uint
	var ret []Distribution

	C.mm_extract_peaks(mm.wrapper, &dist, &count)

	for i := 0; i < int(count); i++ {
		d := C.get_peak(dist, C.uint(i))

		mean := make([]float32, mm.wrapper.dimensions)
		for j := 0; j < int(mm.wrapper.dimensions); j++ {
			mean[j] = float32(C.get_element(d.Mean, C.uint(j)))
		}

		ret = append(ret, Distribution{
			Mean:   mean,
			StdDev: float32(d.standard_deviation),
			Count:  int(d.sample_count),
		})
	}

	C.mm_destroy_peaks(mm.wrapper, dist, count)

	return ret
}

type RGB24 struct {
	Pix    []uint8
	Stride int
	Rect   image.Rectangle
}

type RGB struct {
	R, G, B uint8
}

type RGB24Reader struct {
	Reader io.Reader
	Rect   image.Rectangle
}

func (r *RGB24Reader) ReadRGB24() (*RGB24, error) {
	buf := make([]byte, r.Rect.Dx()*r.Rect.Dy()*3)
	if len(buf) == 0 {
		return nil, fmt.Errorf("cannot read zero pixel image")
	}

	cur := 0
	for {
		if n, err := r.Reader.Read(buf[cur:]); err != nil {
			return nil, err
		} else if n+cur < len(buf) {
			cur += n
		} else {
			break
		}
	}

	return &RGB24{
		Pix:    buf,
		Stride: r.Rect.Dx() * 3,
		Rect:   r.Rect,
	}, nil
}

var RGBModel color.Model = color.ModelFunc(func(c color.Color) color.Color {
	r, g, b, _ := c.RGBA()
	return RGB{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8)}
})

func (c RGB) RGBA() (r, g, b, a uint32) {
	r = uint32(c.R) << 8
	g = uint32(c.G) << 8
	b = uint32(c.B) << 8
	return
}

func NewRGB(r image.Rectangle) *RGB24 {
	return &RGB24{
		Rect:   r.Canon(),
		Stride: 3 * r.Dx(),
		Pix:    make([]uint8, 3*r.Dx()*r.Dy()),
	}
}
func FromImage(img image.Image) *RGB24 {
	if r, ok := img.(*RGB24); ok {
		return r
	}
	// this is really slow for now...
	r := NewRGB(img.Bounds())
	for x := r.Rect.Min.X; x < r.Rect.Max.X; x++ {
		for y := r.Rect.Min.Y; y < r.Rect.Max.Y; y++ {
			r.Set(x, y, img.At(x, y))
		}
	}
	return r
}
func FromRaw(b []byte, stride int) *RGB24 {
	return &RGB24{
		Pix:    b,
		Stride: stride,
		Rect:   image.Rect(0, 0, stride/3, len(b)/stride/3),
	}
}

func (p *RGB24) At(x, y int) color.Color {
	if !(image.Point{x, y}.In(p.Rect)) {
		return RGB{}
	}
	i := p.PixOffset(x, y)
	return RGB{
		p.Pix[i], p.Pix[i+1], p.Pix[i+2],
	}
}
func (p *RGB24) Set(x, y int, c color.Color) {
	if !(image.Point{x, y}.In(p.Rect)) {
		return
	}
	i := p.PixOffset(x, y)
	c1 := RGBModel.Convert(c).(RGB)
	p.Pix[i+0] = uint8(c1.R)
	p.Pix[i+1] = uint8(c1.G)
	p.Pix[i+2] = uint8(c1.B)
}

func (p *RGB24) ColorModel() color.Model {
	return RGBModel
}

func (p *RGB24) SubImage(r image.Rectangle) image.Image {
	r = r.Intersect(p.Rect)
	// If r1 and r2 are Rectangles, r1.Intersect(r2) is not guaranteed to be inside
	// either r1 or r2 if the intersection is empty. Without explicitly checking for
	// this, the Pix[i:] expression below can panic.
	if r.Empty() {
		return &RGB24{}
	}
	// TODO: implement this much faster sub image routine, but this requires image stride
	//   in the C code.  right now just copy the image bytes to a new slice
	// i := p.PixOffset(r.Min.X, r.Min.Y)
	// return &RGB24{
	// 	Pix:    p.Pix[i:],
	// 	Stride: p.Stride,
	// 	Rect:   r,
	// }
	ret := &RGB24{
		Stride: r.Dx() * 3,
		Rect:   image.Rect(0, 0, r.Dx(), r.Dy()),
	}
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			i := p.PixOffset(x, y)
			ret.Pix = append(ret.Pix, p.Pix[i], p.Pix[i+1], p.Pix[i+2])
		}
	}
	return ret
}

// PixOffset returns the index of the first element of Pix that corresponds to
// the pixel at (x, y).
func (p *RGB24) PixOffset(x, y int) int {
	return (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)*3
}

func (p *RGB24) Bounds() image.Rectangle {
	return p.Rect
}

type classifier struct {
	Description string
	Weights     string
	Device      string
	classer     *C.classifier
}
type classifier_response struct {
	Duration  float32
	Embedding []float32
}

func NewClassifier(descriptionFile string, weightsFile string, device string) *classifier {
	return &classifier{
		Description: descriptionFile,
		Weights:     weightsFile,
		Device:      device,
		classer:     C.create_classifier(C.CString(descriptionFile), C.CString(weightsFile), C.CString(device)),
	}
}
func (c *classifier) Close() {
	C.destroy_classifier(c.classer)
}
func (c *classifier) InferRGB24(rgb *RGB24) classifier_response {
	res := C.do_classification(c.classer, unsafe.Pointer(&rgb.Pix[0]), C.int(rgb.Stride),
		C.int(rgb.Rect.Min.X), C.int(rgb.Rect.Min.Y), C.int(rgb.Rect.Max.X), C.int(rgb.Rect.Max.Y))
	// res := C.do_classification_param(c.classer, unsafe.Pointer(&rgb.Pix[0]), C.uint(rgb.Bounds().Dx()), C.uint(rgb.Bounds().Dy()))
	defer C.destroy_classifier_response(res)

	ret := classifier_response{
		Duration: float32(res.duration),
	}

	for i := C.uint(0); i < res.embedding_size; i++ {
		ret.Embedding = append(ret.Embedding, float32(C.get_embedding_at(res.embedding, C.int(i))))
	}

	return ret
}

type detector struct {
	Description string
	Weights     string
	Device      string
	detect      *C.detector
}
type detection struct {
	Confidence float32
	Label      float32
	Rect       image.Rectangle
}

func NewDetector(descriptionFile string, weightsFile string, deviceName string) *detector {
	ret := &detector{
		Description: descriptionFile,
		Weights:     weightsFile,
		Device:      deviceName,
	}
	C.create_face_detector(
		C.CString(descriptionFile),
		C.CString(weightsFile),
		C.CString(deviceName),
		&(ret.detect))
	return ret
}

/*
Close cleans up the memory of the detector.  This must be called to ensure no memory leaks
*/
func (d *detector) Close() {
	C.destroy_face_detector(d.detect)
}

func (d *detector) InferRGB(rgb *RGB24) []detection {
	res := C.do_inference(d.detect, unsafe.Pointer(&rgb.Pix[0]),
		C.int(rgb.Stride), C.int(rgb.Rect.Min.X), C.int(rgb.Rect.Min.Y), C.int(rgb.Rect.Max.X), C.int(rgb.Rect.Max.Y))
	defer C.destroy_response(res)

	var ret []detection

	for i := C.ulong(0); i < res.num_detections; i++ {
		det := C.get_response_detection(res, i)

		x0 := math.Max(float64(det.xmin), 0)
		x1 := math.Min(float64(det.xmax), 1)
		y0 := math.Max(float64(det.ymin), 0)
		y1 := math.Min(float64(det.ymax), 1)

		tec := detection{
			Confidence: float32(det.confidence),
			Label:      float32(det.label),
			Rect: image.Rect(
				int(x0*float64(rgb.Bounds().Dx()))+rgb.Bounds().Min.X,
				int(y0*float64(rgb.Bounds().Dy()))+rgb.Bounds().Min.Y,
				int(x1*float64(rgb.Bounds().Dx()))+rgb.Bounds().Min.X,
				int(y1*float64(rgb.Bounds().Dy()))+rgb.Bounds().Min.Y),
		}

		ret = append(ret, tec)
	}
	return ret
}

var (
	image_width              = 672
	image_height             = 384
	num_channels             = 3
	detectionPadding float32 = 1.275
)

func scaleRectangle(r image.Rectangle, factor float32) image.Rectangle {
	dx := int(float32(r.Dx()) * (factor - 1.0) / 2)
	dy := int(float32(r.Dy()) * (factor - 1.0) / 2)

	return image.Rect(r.Min.X-dx, r.Min.Y-dy, r.Max.X+dx, r.Max.Y+dy)
}

func encodeEmbedding(embedding []float32) string {
	bb := make([]byte, 512/8)
	for i, f := range embedding {
		index := i / 8
		var bit uint = uint(i) % 8

		if f > 0. {
			bb[index] |= 1 << bit
		}
	}
	// log.Printf("hash: %s", base64.StdEncoding.EncodeToString(bb))
	return hex.EncodeToString(bb)
}

func main() {
	det := NewDetector(
		"../detect_faces/face-detection-model/FP16/face-detection-adas-0001.xml",
		"../detect_faces/face-detection-model/FP16/face-detection-adas-0001.bin",
		"MYRIAD")
	defer det.Close()

	classer := NewClassifier(
		"../detect_faces/facenet-model/FP16/20180402-114759.xml",
		"../detect_faces/facenet-model/FP16/20180402-114759.bin",
		"MYRIAD")
	defer classer.Close()

	multiModal := NewMultiModal(512, 1024)
	defer multiModal.Close()

	reader := RGB24Reader{
		Reader: os.Stdin,
		Rect:   image.Rect(0, 0, image_width, image_height),
	}

	item := 0

	lastPeaks := time.Now()
	peakInterval := 1 * time.Minute

	for {
		if rgb, err := reader.ReadRGB24(); err != nil {
			log.Print(err)
			break
		} else {
			detections := det.InferRGB(rgb)

			log.Printf("found: %d", len(detections))

			for _, d := range detections {
				// log.Printf("%d: confidence: %f, (%d %d) - (%d %d)", i, d.Confidence, d.Rect.Min.X, d.Rect.Min.Y, d.Rect.Max.X, d.Rect.Max.Y)

				// padd the rectangle to get more of the face
				r := scaleRectangle(d.Rect, detectionPadding)

				if !r.In(reader.Rect) {
					// out of bounds
					continue
				}

				face := rgb.SubImage(r)

				if f, err := os.OpenFile(fmt.Sprintf("faces/face%05d.jpg", item), os.O_CREATE|os.O_TRUNC|os.O_RDWR, 0660); err != nil {
					log.Print(err)
					return
				} else {
					jpeg.Encode(f, face, &jpeg.Options{90})
					item = (item + 1) % 1024
				}

				classification := classer.InferRGB24(face.(*RGB24))
				multiModal.Insert(classification.Embedding)

				// log.Printf("classification duration: %f ms", classification.Duration)
				// log.Printf("embedding: %v", classification.Embedding)

				// log.Printf("hash: %s", base64.StdEncoding.EncodeToString(bb))
				log.Printf("hex: %s", encodeEmbedding(classification.Embedding))

				if time.Now().Sub(lastPeaks) > peakInterval {
					lastPeaks = time.Now()
					peaks := multiModal.Peaks()

					log.Printf("number of people found: %d", len(peaks))
					for _, p := range peaks {
						log.Printf("%d %f %s", p.Count, p.StdDev)
						log.Printf("hex: %s", encodeEmbedding(p.Mean))
					}

					peakBytes := json.Marshal(peaks)
					ioutil.WriteFile("peaks.json", peakBytes, 0664)
				}

			}
		}
	}
}
