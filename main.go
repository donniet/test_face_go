package main

import (
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/jpeg"
	"io/ioutil"
	"log"
	"os"
	"time"

	detector "github.com/donniet/detector"
)

var (
	image_width              = 1920
	image_height             = 1088
	num_channels             = 3
	detectionPadding float32 = 1.275
	maxError         float32 = 0.15
	minSamples               = 500
	saveFile                 = "faces.multimodal"
	device                   = "MYRIAD"
)

func init() {
	flag.StringVar(&device, "device", device, "device name")
}

func scaleRectangle(r image.Rectangle, factor float32) image.Rectangle {
	dx := int(float32(r.Dx()) * (factor - 1.0) / 2)
	dy := int(float32(r.Dy()) * (factor - 1.0) / 2)

	return image.Rect(r.Min.X-dx, r.Min.Y-dy, r.Max.X+dx, r.Max.Y+dy)
}

func encodeEmbedding(embedding []float32) string {
	bb := make([]byte, len(embedding)/8)
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

// func standardize(rgb *detector.RGB24) []float32 {
// 	// calculate mean
// 	sum := 0.
// 	for _, p := range rgb.Pix {
// 		sum += float64(p)
// 	}
// 	mean := sum / float64(len(rgb.Pix))
// 	std := 0.

// 	for _, p := range rgb.Pix {
// 		t := float64(p) - mean
// 		std += t * t
// 	}
// 	std = math.Sqrt(std)
// 	std_adj := math.Max(std, 1./math.Sqrt(float64(len(rgb.Pix))))

// 	ret := make([]float32, len(rgb.Pix))
// 	for i, p := range rgb.Pix {
// 		ret[i] = float32((float64(p) - mean) / std_adj)
// 	}
// 	return ret
// }

func main() {
	flag.Parse()

	home := "/home/donniet"
	home, _ = os.LookupEnv("HOME")

	var det *detector.Detector
	var classer *detector.Classifier

	if device == "MYRIAD" {
		det = detector.NewDetector(
			home+"/src/detect_faces/face-detection-model/FP16/face-detection-adas-0001.xml",
			home+"/src/detect_faces/face-detection-model/FP16/face-detection-adas-0001.bin",
			"MYRIAD")
		// classer = detector.NewClassifier(
		// 	home+"/src/detect_faces/facenet-model/FP16/20180402-114759.xml",
		// 	home+"/src/detect_faces/facenet-model/FP16/20180402-114759.bin",
		// 	"MYRIAD")
		// classer = detector.NewClassifier(
		// 	home+"/Downloads/20180402-114759/facenet_fp16.xml",
		// 	home+"/Downloads/20180402-114759/facenet_fp16.bin",
		// 	"MYRIAD")
		classer = detector.NewClassifier(
			home+"/src/detect_faces/resnet50_128_caffe/FP16/resnet50_128.xml",
			home+"/src/detect_faces/resnet50_128_caffe/FP16/resnet50_128.bin",
			"MYRIAD")
	} else {
		det = detector.NewDetector(
			home+"/src/detect_faces/face-detection-model/FP32/face-detection-adas-0001.xml",
			home+"/src/detect_faces/face-detection-model/FP32/face-detection-adas-0001.bin",
			"CPU")
		// classer = detector.NewClassifier(
		// 	home+"/Downloads/20180402-114759/facenet.xml",
		// 	home+"/Downloads/20180402-114759/facenet.bin",
		// 	"CPU")
		// classer = detector.NewClassifier(
		// 	home+"/src/detect_faces/facenet-model/FP32/20180402-114759.xml",
		// 	home+"/src/detect_faces/facenet-model/FP32/20180402-114759.bin",
		// 	"CPU")
		// classer = detector.NewClassifier(
		// 	home+"/src/Face-Recognition-with-OpenVino-Toolkit/model/20180402-114759.xml",
		// 	home+"/src/Face-Recognition-with-OpenVino-Toolkit/model/20180402-114759.bin",
		// 	"CPU")
		classer = detector.NewClassifier(
			home+"/src/detect_faces/resnet50_128_caffe/FP32/resnet50_128.xml",
			home+"/src/detect_faces/resnet50_128_caffe/FP32/resnet50_128.bin",
			"CPU")
	}
	defer det.Close()
	defer classer.Close()

	multiModal := detector.NewMultiModal(128, 1024)

	if f, err := os.OpenFile(saveFile, os.O_RDONLY, 0660); err != nil {
		log.Printf("error opening save file: %v, continuing empty", err)
	} else if _, err := multiModal.ReadFrom(f); err != nil {
		log.Fatal(err)
	} else {
		f.Close()
	}

	defer multiModal.Close()

	reader := detector.RGB24Reader{
		Reader: os.Stdin,
		Rect:   image.Rect(0, 0, image_width, image_height),
	}

	item := 0

	lastPeaks := time.Now()
	peakInterval := 1 * time.Minute

	// scaler := draw.ApproxBiLinear

	for {
		if rgb, err := reader.ReadRGB24(); err != nil {
			log.Print(err)
			break
		} else {

//                        if f, err := os.OpenFile("frame.jpg", os.O_CREATE|os.O_TRUNC|os.O_RDWR, 0660); err != nil {
//                                log.Print(err)
//                                return
//                        } else {
//                                jpeg.Encode(f, rgb, &jpeg.Options{90})
//
//                        }

					
			
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

				// // assume 160x160 for now, but get from the classer later
				// scaled := detector.NewRGB(image.Rect(0, 0, 160, 160))

				// log.Printf("scaling %s to %s", face.Bounds(), scaled.Bounds())
				// scaler.Scale(scaled, scaled.Bounds(), face, face.Bounds(), draw.Over, nil)
				// // draw.Draw(scaled, scaled.Bounds(), face, face.Bounds().Min, draw.Over)

				classification := classer.InferRGB24(face.(*detector.RGB24))
				multiModal.Insert(classification.Embedding)

				dist := multiModal.Find(classification.Embedding)

				embeddingBytes, _ := json.Marshal(classification.Embedding)

				os.MkdirAll("faces/unknown", 0770)
				jpegName := fmt.Sprintf("faces/unknown/face%05d.jpg", item)
				embeddingName := fmt.Sprintf("faces/unknown/face%05d.json", item)

				erf := dist.Erf(classification.Embedding)

				log.Printf("nearest dist: %d prob %f", dist.Id, erf)

				if dist.Count > minSamples && erf < maxError {
					os.MkdirAll(fmt.Sprintf("faces/%d", dist.Id), 0770)

					jpegName = fmt.Sprintf("faces/%d/face%05d.jpg", dist.Id, item)
					embeddingName = fmt.Sprintf("faces/%d/face%05d.json", dist.Id, item)
				}

				if f, err := os.OpenFile(jpegName, os.O_CREATE|os.O_TRUNC|os.O_RDWR, 0660); err != nil {
					log.Print(err)
					return
				} else {
					jpeg.Encode(f, face, &jpeg.Options{90})
					item = (item + 1) % 100000
				}

				ioutil.WriteFile(embeddingName, embeddingBytes, 0664)

				// log.Printf("classification duration: %f ms", classification.Duration)
				// log.Printf("embedding: %v", classification.Embedding)

				// log.Printf("hash: %s", base64.StdEncoding.EncodeToString(bb))
				log.Printf("hex: %s", encodeEmbedding(classification.Embedding))

				if time.Now().Sub(lastPeaks) > peakInterval {
					lastPeaks = time.Now()
					peaks := multiModal.Peaks()

					log.Printf("number of people found: %d", len(peaks))
					for _, p := range peaks {
						log.Printf("%d %f %d", p.Count, p.StdDev, p.Id)
						log.Printf("hex: %s", encodeEmbedding(p.Mean))
					}

					peakBytes, _ := json.Marshal(peaks)
					ioutil.WriteFile("peaks.json", peakBytes, 0664)

					// writing out datastructure
					if f, err := os.OpenFile(saveFile, os.O_TRUNC|os.O_CREATE|os.O_WRONLY, 0660); err != nil {
						log.Fatal(err)
					} else if _, err := multiModal.WriteTo(f); err != nil {
						log.Fatal(err)
					} else {
						f.Close()
					}
				}

			}
		}
	}
}
