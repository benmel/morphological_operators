morphological_operators
================

- Applies morphological operators to binary images
- Morphological operators are used to find boundaries in images and remove noise
- A structuring element is used to analyze pixels in a certain area
- Each operator has a different effect on the input image
- Eight choices of operators
  * `erosion`: bright areas grow, dark areas shrink
  * `dilation`: bright areas shrink, dark areas grow
  * `opening`: erosion followed by dilation
  * `closing`: dilation followed by erosion
  * `boundary`: eroded image subtracted from input image
  * `closing_boundary`: closing followed by boundary
  * `closing_opening`: closing followed by opening
  * `closing_opening_boundary`: closing followed by opening followed by boundary
- Eight choices of structuring elements: `rect3x3`, `rect5x5`, `rect7x7`, `rect9x9`, `rect11x11`, `rect13x13`, `disk6` or `disk10`
- For example `rect3x3` is a 3 by 3 rectangle and `disk6` is a disk with radius 6
- Usage: `python morphological_operators.py -m <morph_op> -s <se> -i <inputf> [-o <outputf>]`
- Optional `-o` flag specifies output file location
- Example: `python morphological_operators.py -m opening -s disk6 -i images/palm.bmp -o results/palm.png`

Input Image

![alt tag](http://i.imgur.com/9Xecrig.png)

Results

Operator | Structuring Element |  Output Image
-------- | ------------------- |  ------------
`erosion` | `rect3x3` | ![alt tag](http://i.imgur.com/1rZ9tbc.png)
`dilation` | `rect3x3` | ![alt tag](http://i.imgur.com/gtGiwKA.png)
`opening` | `rect3x3` | ![alt tag](http://i.imgur.com/NDSOydS.png)
`closing` | `rect3x3` | ![alt tag](http://i.imgur.com/5yhB195.png)
`closing` | `disk10` | ![alt tag](http://i.imgur.com/RcWX59Y.png)
`boundary` | `rect3x3` | ![alt tag](http://i.imgur.com/zEt8Vdj.png)
`closing_boundary` | `disk10` | ![alt tag](http://i.imgur.com/tJk3nKT.png)
