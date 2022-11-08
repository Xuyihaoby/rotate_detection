# About angle definition

## version: 'v1', ('oc' in mmrotate) $[-\pi/2, 0)$

you must notice that, cv2.minAreaRect() is different between before opencv=4.5.1 and after opencv==4.5.1.

Before 4.5.1, the function definition follows the guidelines about:

<img src="./About_angle_definition.assets/image-20221108171500707.png"/>

After 4.5.1, the function definition follows the guidelines ablout:

<img src="./About_angle_definition.assets/image-20221108173351845.png"/>

## version: 'v2', ('le135' in mmrotate) $[-\pi/4, 3\pi/4 )$

<img src="./About_angle_definition.assets/image-20221108215406484.png"/>

## version: 'v3', ('le90' in mmrotate) $(-\pi/2, \pi/2]$

<img src="./About_angle_definition.assets/image-20221108191856052.png"/>

### note

In version 'v1' and 'v2', the angle follow the left hand coordinate system, 'v3' is right hand coordinate system