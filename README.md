# HDR imaging
## ( Modified from https://github.com/SSARCandy/HDR-imaging )


## Requirement

- C++ 11
- opencv 3.0 (or higher)
- lapack and lapacke

## Usage

```bash
$ make clean && make
$ ./hdr <input img dir> <output .hdr name>

# for example
$ ./hdr taipei taipei.hdr
```

## Input format

The input dir should have:

- Some `.png` images
- A `image_list.txt`, file should contain:
  - filename
  - exposure
  - 1/shutter_speed
  - number of images in the folder ( at the very first line )

This is an example for `image_list.txt`:

```
10
# Filename   exposure 1/shutter_speed
DSC_0058.png 32        0.03125
DSC_0059.png 16        0.0625
DSC_0060.png  8        0.125
DSC_0061.png  4        0.25
DSC_0062.png  2        0.5
DSC_0063.png  1        1
DSC_0064.png  0.5      2
DSC_0065.png  0.25     4
DSC_0066.png  0.125    8
DSC_0067.png  0.0625  16
```

## Output

The program will output:

- A `.hdr` image

## Environment

I test my code in Rock64 with Armbian 5.65 ( ubuntu 18.04 based ), but it should work fine in macOS/Linux/Windows ( maybe with some header swap to match opencv header on different OS )
