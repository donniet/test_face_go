package main

import (
	"encoding/hex"
	"encoding/json"
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
	image_width              = 672
	image_height             = 384
	num_channels             = 3
	detectionPadding float32 = 1.275
	maxError         float32 = 0.3
	minSamples               = 500
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
	det := detector.NewDetector(
		"../detect_faces/face-detection-model/FP16/face-detection-adas-0001.xml",
		"../detect_faces/face-detection-model/FP16/face-detection-adas-0001.bin",
		"MYRIAD")
	defer det.Close()

	classer := detector.NewClassifier(
		"../detect_faces/facenet-model/FP16/20180402-114759.xml",
		"../detect_faces/facenet-model/FP16/20180402-114759.bin",
		"MYRIAD")
	defer classer.Close()

	multiModal := detector.NewMultiModal(512, 1024)
	defer multiModal.Close()

	reader := detector.RGB24Reader{
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
				}

			}
		}
	}
}
