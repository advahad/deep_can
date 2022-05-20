# DeepCAN: Hybrid Method for Road Type Classification Using Vehicle Sensor Data for Smart Autonomous Mobility
Code implementation of paper: DeepCAN: Hybrid Method for Road Type Classification Using Vehicle Sensor Data for Smart Autonomous Mobility
SHRP2 dataset' sample can be found here: https://insight.shrp2nds.us/
The european dataset is proprietary and cannot be shared due to privacy issues


Usually, image- and radar-based data are used
to perform environmental characteristics related tasks in au-
tonomous cars, while the use of sensor data from the Controller
Area Network (CAN) bus has been limited. We explore the
use of this valuable sensor data. The vehicle’s CAN bus data
consist of multivariate time series data, such as velocity, RPM,
and acceleration, which contain meaningful information about
the vehicle dynamics and environmental characteristics. The
ability to use these data to sense the environment, along with
the sight sense (from image-based data), can prevent a single
point of failure when image- or radar-based data are missing
or incomplete, and contribute to increased understanding of
the vehicle’s environment. A solution that does not rely on
image- or radar-based data also addresses concerns about privacy
and the use of location-based data. We present DeepCAN, a
novel hybrid method for road type classification that utilizes
solely vehicle dynamics data and combines two main approaches
for time series classification. The end-to-end approach uses a
fully convolutional network which is extended with latent long-
term feature representation, while the feature-based approach
utilizes an XGBoost classifier with aggregated time series feature
representation. In our comprehensive evaluation on two real-
world datasets, the performance of each model component was
assessed separately as an independent solution, along with the
performance of a model integrating all of the components in
a hybrid solution. The results demonstrate the efficiency and
accuracy of DeepCAN and provide a solid basis for its future
use by the automobile industry.
Index Terms—deep learning, GBM, XGBoost, FCN, road
type classification, time series, sensors, CAN bus, autonomous
mobility.
