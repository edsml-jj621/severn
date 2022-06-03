##########
Flood Tool
##########

This package implements a flood risk prediction and visualization tool.

This document contains the latest information about Flood Tool package, which is automatically generated on the remote Linux virtual machine hosted by GitHub. 

Installation Instructions
-------------------------

To install the module ``flood_tools`` clone the respository to local by running:

``git clone https://github.com/ese-msc-2021/ads-deluge-severn.git``.

Then navigateto the root of the repository that you cloned and install all the packages needed by
running:

``pip install -r requirements.txt``

``pip install -e .``.


Usage guide
-----------

``python flood_tool/main.py [-h] -t LABEL_TYPE -f UNLABELLED_FILE [-m METHOD] [-s SAVE]``

+--------------------------+--------------------------------------------------+
|          Options         | Description                                      |
+==========================+==================================================+
|        -h, --help        | help message                                     |
+--------------------------+--------------------------------------------------+
|                          | Type of labelling.                               |
|      -t, --label_type    | | -t flood_risk                                  |
|                          | | -t house_price                                 |
+--------------------------+--------------------------------------------------+
|                          | Unlabelled postcodes file.                       |
|      -f postcodes.csv    | | -f postcodes.csv                               |
|                          |                                                  |
+--------------------------+--------------------------------------------------+
|                          | (optional)                                       |
|                          |                                                  |
|                          | Flood Risk                                       |
|                          | | *default knn*                                  |
|                          | | -m dt:  Decision Tree                          |
|                          | | -m knn:  KNN                                   |
|                          | | -m rmdf:  Random Forest                        |
|                          | | -m ada:  AdaBoost                              |
|    -m METHOD, --method   |                                                  |
|                          | House Price                                      |
|                          | | *default rfr*                                  |
|                          | | -m lr:  Linear Regression                      |
|                          | | -m dt:  Dscision Tree                          |
|                          | | -m rfr: Random Forest Regression               |
|                          | | -m sv:  SVR Support Vector Regression          |
|                          |                                                  |
+--------------------------+--------------------------------------------------+
|                          | (optional)                                       |
|      -s, --save          | *default labelled.csv*                           |
|                          | output filename / path                           |
+--------------------------+--------------------------------------------------+


Geodetic Transformations
------------------------

For historical reasons, multiple coordinate systems exist in in current use in
British mapping circles. The Ordnance Survey has been mapping the British Isles
since the 18th Century and the last major retriangulation from 1936-1962 produced
the Ordance Survey National Grid (otherwise known as **OSGB36**), which defined
latitude and longitude for all points across the island of Great Britain [1]_.
For convenience, a standard Transverse Mercator projection [2]_ was also defined,
producing a notionally flat 2D gridded surface, with gradations called eastings
and northings. The scale for these gradations was identified with metres, which
allowed local distances to be defined with a fair degree of accuracy.


The OSGB36 datum is based on the Airy Ellipsoid of 1830, which defines
semimajor axes for its model of the earth, :math:`a` and :math:`b`, a scaling
factor :math:`F_0` and ellipsoid height, :math:`H`.

.. math::
    a &= 6377563.396, \\
    b &= 6356256.910, \\
    F_0 &= 0.9996012717, \\
    H &= 24.7.

The point of origin for the transverse Mercator projection is defined in the
Ordnance Survey longitude-latitude and easting-northing coordinates as

.. math::
    \phi^{OS}_0 &= 49^\circ \mbox{ north}, \\
    \lambda^{OS}_0 &= 2^\circ \mbox{ west}, \\
    E^{OS}_0 &= 400000 m, \\
    N^{OS}_0 &= -100000 m.

More recently, the world has gravitated towards the use of satellite based GPS
equipment, which uses the (globally more appropriate) World Geodetic System
1984 (also known as **WGS84**). This datum uses a different ellipsoid, which offers a
better fit for a global coordinate system (as well as North America). Its key
properties are:

.. math::
    a_{WGS} &= 6378137,, \\
    b_{WGS} &= 6356752.314, \\
    F_0 &= 0.9996.

For a given point on the WGS84 ellipsoid, an approximate mapping to the
OSGB36 datum can be found using a Helmert transformation [3]_,

.. math::
    \mathbf{x}^{OS} = \mathbf{t}+\mathbf{M}\mathbf{x}^{WGS}.


Here :math:`\mathbf{x}` denotes a coordinate in Cartesian space (i.e in 3D)
as given by the (invertible) transformation

.. math::
    \nu &= \frac{aF_0}{\sqrt{1-e^2\sin^2(\phi^{OS})}} \\
    x &= (\nu+H) \sin(\lambda)\cos(\phi) \\
    y &= (\nu+H) \cos(\lambda)\cos(\phi) \\
    z &= ((1-e^2)\nu+H)\sin(\phi)

and the transformation parameters are

.. math::
    :nowrap:

    \begin{eqnarray*}
    \mathbf{t} &= \left(\begin{array}{c}
    -446.448\\ 125.157\\ -542.060
    \end{array}\right),\\
    \mathbf{M} &= \left[\begin{array}{ c c c }
    1+s& -r_3& r_2\\
    r_3 & 1+s & -r_1 \\
    -r_2 & r_1 & 1+s
    \end{array}\right], \\
    s &= 20.4894\times 10^{-6}, \\
    \mathbf{r} &= [0.1502'', 0.2470'', 0.8421''].
    \end{eqnarray*}

Given a latitude, :math:`\phi^{OS}` and longitude, :math:`\lambda^{OS}` in the
OSGB36 datum, easting and northing coordinates, :math:`E^{OS}` & :math:`N^{OS}`
can then be calculated using the following formulae (see "A guide to coordinate
systems in Great Britain, Appendix C1):

.. math::
    \rho &= \frac{aF_0(1-e^2)}{\left(1-e^2\sin^2(\phi^{OS})\right)^{\frac{3}{2}}} \\
    \eta &= \sqrt{\frac{\nu}{\rho}-1} \\
    M &= bF_0\left[\left(1+n+\frac{5}{4}n^2+\frac{5}{4}n^3\right)(\phi^{OS}-\phi^{OS}_0)\right. \\
    &\quad-\left(3n+3n^2+\frac{21}{8}n^3\right)\sin(\phi-\phi_0)\cos(\phi^{OS}+\phi^{OS}_0) \\
    &\quad+\left(\frac{15}{8}n^2+\frac{15}{8}n^3\right)\sin(2(\phi^{OS}-\phi^{OS}_0))\cos(2(\phi^{OS}+\phi^{OS}_0)) \\
    &\left.\quad-\frac{35}{24}n^3\sin(3(\phi-\phi_0))\cos(3(\phi^{OS}+\phi^{OS}_0))\right] \\
    I &= M + N^{OS}_0 \\
    II &= \frac{\nu}{2}\sin(\phi^{OS})\cos(\phi^{OS}) \\
    III &= \frac{\nu}{24}\sin(\phi^{OS})cos^3(\phi^{OS})(5-\tan^2(phi^{OS})+9\eta^2) \\
    IIIA &= \frac{\nu}{720}\sin(\phi^{OS})cos^5(\phi^{OS})(61-58\tan^2(\phi^{OS})+\tan^4(\phi^{OS})) \\
    IV &= \nu\cos(\phi^{OS}) \\
    V &= \frac{\nu}{6}\cos^3(\phi^{OS})\left(\frac{\nu}{\rho}-\tan^2(\phi^{OS})\right) \\
    VI &= \frac{\nu}{120}\cos^5(\phi^{OS})(5-18\tan^2(\phi^{OS})+\tan^4(\phi^{OS}) \\
    &\quad+14\eta^2-58\tan^2(\phi^{OS})\eta^2) \\
    E^{OS} &= E^{OS}_0+IV(\lambda^{OS}-\lambda^{OS}_0)+V(\lambda-\lambda^{OS}_0)^3+VI(\lambda^{OS}-\lambda^{OS}_0)^5 \\
    N^{OS} &= I + II(\lambda^{OS}-\lambda^{OS}_0)^2+III(\lambda-\lambda^{OS}_0)^4+IIIA(\lambda^{OS}-\lambda^{OS}_0)^6

The inverse transformation can be generated iteratively using a fixed point process:

1. Set :math:`M=0` and :math:`\phi^{OS} = \phi_0^{OS}`.
2. Update :math:`\phi_{i+1}^{OS} = \frac{N-N_0-M}{aF_0}+\phi_i^{OS}`
3. Calculate :math:`M` using the formula above.
4. If :math:`\textrm{abs}(N-N_0-M)> 0.01 mm` go to 2, otherwise halt.

With :math:`M` calculated we now improve our estimate of :math:`\phi^{OS}`. First calculate
:math:`\nu`, :math:`\rho` and :math:`\eta` using our previous formulae. Next

.. math::

    VII &= \frac{\tan(\phi^{OS})}{2\rho\nu},\\
    VIII &= \frac{\tan(\phi^{OS})}{24\rho\nu^3}\left(5+3\tan^2(\phi^{OS})+\eta^2-9\tan^2(\phi^{OS})\eta^2\right),\\
    IX &= \frac{\tan(\phi^{OS})}{720\rho\nu^5}\left(61+90\tan^2(\phi^{OS})+45\tan^4(\phi^{OS})\right),\\
    X &= \frac{\sec\phi^{OS}}{\nu}, \\
    XI &= \frac{\sec\phi^{OS}}{6\nu^3}\left(\frac{\nu}{\rho}+2\tan^2(\phi^{OS})\right), \\
    XII &= \frac{\sec\phi^{OS}}{120\nu^5}\left(5+28\tan^2(\phi^{OS})+24\tan^4(\phi^{OS})\right), \\
    XIIA &= \frac{\sec\phi^{OS}}{5040\nu^5}\left(61+662\tan^2(\phi^{OS})+1320\tan^4(\phi^{OS})+720\tan^6(\phi^{OS})\right).

Finally, the corrected values for :math:`\phi^{OS}` and :math:`\lambda^{OS}` are:

.. math::
    \phi_{\textrm{final}}^{OS} &= \phi^{OS} -VII(E-E_0)^2 +VIII(E-E_0)^4 -IX(E-E_0)^6, \\
    \lambda_{\textrm{final}}^{OS} &= \lambda_0^{OS}+X(E-E_0)-XI(E-E_0)^3+ XII(E-E_0)^5-XII(E-E_0)^7.


Classifier choice
-----------------

In order to give more options for who would use this module to make predictions about the flood probability, we provided several trained classifiers:

- Decision Tree
- K-Nearest Neighbors
- Random Forest
- AdaBoost

:Decision Tree: The decision tree classifier (Pang-Ning et al., 2006) creates the classification model by building a decision tree.Each node in the tree specifies a test on an attribute, each branch descending from that node corresponds to one of the possible values for that attribute.

:K-Nearest Neighbors: In statistics, the k-nearest neighbors algorithm (k-NN) is a non-parametric classification method first developed by Evelyn Fix and Joseph Hodges in 1951, and later expanded by Thomas Cover. It is used for classification and regression. In both cases, the input consists of the k closest training examples in a data set.

:Random Forest: As the name suggests, "Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset." Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output.

:AdaBoost: An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.

Regressor choice
----------------
:Just like the classifier choice, we provided many trained regressors for users as well:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Supprot Vector Regressor

:Linear Regression: In statistics, linear regression is a linear approach for modelling the relationship between a scalar response and one or more explanatory variables. The case of one explanatory variable is called simple linear regression; for more than one, the process is called multiple linear regression.

:Decision Tree Regressor: Decision tree builds regression or classification models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes.

:Random Forest Regressor: Random Forest Regression is a supervised learning algorithm that uses ensemble learning method for regression. Ensemble learning method is a technique that combines predictions from multiple machine learning algorithms to make a more accurate prediction than a single model.

:Supprot Vector Regressor: Support Vector Regression is a supervised learning algorithm that is used to predict discrete values. Support Vector Regression uses the same principle as the SVMs. The basic idea behind SVR is to find the best fit line. In SVR, the best fit line is the hyperplane that has the maximum number of points.


Data visualization
----------------

Fig.1 shows all the information we have about the given postcode using a popup message (a html table). It shows the total rainfall in mm and maximum river level in mASD of that location on a wet day (you can specify a typical day if you want in the code). Both of rainfall and river level data are retrieved from the closest monitoring station from that postcode. And it shows the maximum rainfall class of the day. The rainfall classifier is based on the table shown below. The following information, flood event probability, property value, and flood risk are all predicted by our model. 

    .. figure:: pic1.png
            :width: 500px
            :align: center
            :alt: alternate text
            :figclass: align-center

            Indivitual postcode location with corresponding Flood and rainfall information

Fig.2 visualizes the sreading of predicted flood probability among the selected postcodes from the test file. The flood event probabilities for the areas were generated from the riskLabel classifier defined in tool.py. The color map was chosen as perceptional sequential color range from orange to dark red(which representes low to high value) with a sequential gradient. The individual probability of flood for each postcode could be displayed after a mouse click. The map shows that north-eastern coastal area of the UK has a generally higher flood event probability than western area. There are a few darker points showing the riskest areas of flood including Windsor, Silsden, Gainsborough and etc. They have been observed with a common feature that they are located next to a river or they are in costal areas. 

    .. figure:: pic2.png
            :width: 500px
            :align: center
            :alt: alternate text
            :figclass: align-center

            The spreading of predicted flood probability among the selected postcode

Fig.3 and Fig.4 visualizes the heat map for maximum daily rainfall and riverlevels at a specific date. The first map shows the rainfall heatmap and the second map shows the riverlevel heatmap. The color map was chosen as perceptional sequential color from lime to blue(low to high) in the rainfall heatmap. The rainfall heatmap shows that the rainfall in the northern part around Leads and Manchester is higher than other parts indicated by the darker color. In the riverlevel heatmap, the map shows that the riverlevel is higher in Southern Uk near. These patterns match the flood probability plot that higher flood risk occurs in Northeast and Southeast. 

    .. figure:: pic3.png
            :width: 500px
            :align: center
            :alt: alternate text
            :figclass: align-center

            The heatmap for the total daily rainfall and riverlevel for the selected postcodes(1)

    .. figure:: pic4.png
            :width: 500px
            :align: center
            :alt: alternate text
            :figclass: align-center

            The heatmap for the total daily rainfall and riverlevel for the selected postcodes(2)

Fig.5 visualizes the rainfall heatmap variation in a 24 hour period on 5th May 2021. As we go through the animation, we can see that there is little rainfall variation in the South-eastern part of the UK among the 24 hours timeline, while the rainfall patterns in the North-eastern have a large rainfall variation in the same period. This means the reliability of rainfall prediction in the North is lower than other areas, as its'rapid rainfall variations could result in uncerntainties in its rainfall pattern estimation. This could be explained by the relatively extreme weather in high latitude area of the UK. We can also see from the animation that the rainfall in the mid day is lower than in the morning and evening. 

    .. figure:: pic5.png
            :width: 500px
            :align: center
            :alt: alternate text
            :figclass: align-center

            The heatmap variation animation with a timeline for 24 hours

Fig.6 shows the predicted property value for the selected postcode (in postcodes_unlabelled.csv) around the UK. There are four categories, including 0-250k, 250k-375k, 375k-500k, and larger than 500k, corresponding to four different colors (light blue, cadet blue, blue and dark blue) respectively. It is found that the distribution and trend are very clear. From south to north, the property value is decreasing gradually in general. Most of the properties near the Great London area larger than 500k (dark blue). The other major cities like Birmingham and Nottingham in the middle of the UK have the property value around 250-375k (cadet blue). As for the north of the UK like Newcastle, Leeds and Manchester, the property value in this area is below 250k (light blue). 

    .. figure:: pic6.png
            :width: 500px
            :align: center
            :alt: alternate text
            :figclass: align-center

            The spread of predicted house price for the selected postcodes

Fig.7 shows the property value in the sample locations given in postcodes_sampled.csv around the UK. The mapping strategy is the same as the map showing the predicted property value. Four different colors (light blue, cadet blue, blue and dark blue) shows four categories. The general distribution and trend are similar to our predicted one. However, as the locations is much more than the predicted one, there are more data points providing the property values. It is found some property values are below 500k in the south of the UK, while some are above 500k in the north of the UK, although only account for a small number of it.

    .. figure:: pic7.png
            :width: 500px
            :align: center
            :alt: alternate text
            :figclass: align-center

            The spreading of median house price in sampled csv

Fig.8 shows the actual flood event probability in the sample locations given in postcodes_sampled.csv around the UK. The mapping strategy is the same as the map showing the predicted property value.This map shows that high flood probability occurs near the north-eartern coastal area of the UK, which shows a similar pattern as the predicted flood event probability map using unlabelled data. 

    .. figure:: pic8.png
            :width: 500px
            :align: center
            :alt: alternate text
            :figclass: align-center

            The spreading of flood probability given in sampled csv


Data analysis
-------------

For the flood class prediction, the x axis of the histogram means 10 different flood classes from 1 to 10. 1 indicates it has only 0.01% probability that the place encounters flood, while class 10 show 5% probability. So the lowest class 1 expects one event in 1000 years (or longer) and the highest risk class 10 expects one event in 20 years (or sooner). Most places are predicted a flood could hardly appear. Several postcode areas have class 6-9, and we should pay more attention to these areas' river and rainfall data.

    .. figure:: pic9.png
            :width: 500px
            :align: center
            :alt: alternate text
            :figclass: align-center

            The predicted flood class distribution

The x axis shows the predicted median house values, and the y axis indicate the number of properties. Using decision tree and random forest methods, the distribution of predicted median price is more divergent, the highest price reach 7e6 and 5e6 recpectively, while through linear regression and sv regressors, the predicted value is in the range of 100000 to 700000 pounds.

    .. figure:: pic10.png
            :width: 500px
            :align: center
            :alt: alternate text
            :figclass: align-center

            The predicted property value distribution

Combining the above three figures, we find the distribution of annual flood risk has the similar pattern with the flood class distribution. Hence, we conclude that the flood risk is mainly decided by the flood class.

    .. figure:: pic11.png
            :width: 500px
            :align: center
            :alt: alternate text
            :figclass: align-center

            The predicted annual flood risk distribution


Function APIs
-------------

.. automodule:: flood_tool
  :members:
  :imported-members:


.. rubric:: References

.. [1] A guide to coordinate systems in Great Britain, Ordnance Survey
.. [2] Map projections - A Working Manual, John P. Snyder, https://doi.org/10.3133/pp1395
.. [3] Computing Helmert transformations, G Watson, http://www.maths.dundee.ac.uk/gawatson/helmertrev.pdf
