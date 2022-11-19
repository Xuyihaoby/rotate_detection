# G-Rep: Gaussian Representation for Arbitrary-Oriented Object Detection

[G-Rep: Gaussian Representation for Arbitrary-Oriented Object Detection](http://arxiv.org/abs/2205.11796)

## Abstract

Arbitrary-oriented object representations contain the oriented bounding box (OBB), quadrilateral bounding box (QBB), and point set (PointSet). Each representation encounters problems that correspond to its characteristics, such as the boundary discontinuity, square-like problem, representation ambiguity, and isolated points, which lead to inaccurate detection. Although many effective strategies have been proposed for various representations, there is still no uniﬁed solution. Current detection methods based on Gaussian modeling have demonstrated the possibility of breaking this dilemma; however, they remain limited to OBB. To go further, in this paper, we propose a uniﬁed Gaussian representation called G-Rep to construct Gaussian distributions for OBB, QBB, and PointSet, which achieves a uniﬁed solution to various representations and problems. Speciﬁcally, PointSet or QBB-based objects are converted into Gaussian distributions, and their parameters are optimized using the maximum likelihood estimation algorithm. Then, three optional Gaussian metrics are explored to optimize the regression loss of the detector because of their excellent parameter optimization mechanisms. Furthermore, we also use Gaussian metrics for sampling to align label assignment and regression loss. Experimental results on several public available datasets, DOTA, HRSC2016, UCAS-AOD, and ICDAR2015 show the excellent performance of the proposed method for arbitrary-oriented object detection.

## Results and Models

| Method | Backbone | Angle     | Loss | Lr schd | Dataset         | preprocess    | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ------ | -------- | --------- | ---- | ------- | --------------- | ------------- | ---------- | ----------- | ----- |
| G-Rep  | ResNet50 | point set | KLD  | 1x      | DOTA-v1.0,train | 1024x1024,512 | 65.68      | \           | \     |

note: the assigner in this version is just ATSS.