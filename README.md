# Sartorius-Cell-Instance-Segmentation

Use Deep Learning models to segment cells in microscopy image.

The dataset is downloaded from: https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/data

**Sample image in this dataset**:

![Microscopy image sample in dataset](https://storage.googleapis.com/kagglesdsdata/competitions/30201/2750748/train/0030fd0e6378.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20220711%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220711T081747Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=6bed3ab9073f9a3c42c1749bc9f80aa10c8b431becf5a2e18a937508f27d0c3146973ba61f90e60be18ee8f658dc82fb8460833a5f12b81b5a9275a2428418e54d7df3bee51877365d59b76560d68c5feec359047c6a1a911e832c6f8c0021e3fc891c0602f613411f5fb623510f2c9bdfbe148c2d1b9273c2425e28b1f0f4265e4442a77b9ca88becb8be7b76d917ec624ac2d27d758c807da2cf5f8344b321bf914857db2c9d471d1ed9ef7fe9240480366226c4b2bad925885e1fbcf6f6194f9107f19123b9bb56064fb8f3d1f8d4a23e9049a1d1df318ae2f0ec8558a02581f16b358228eed40e2fe94f7f2e91ebbc1bbf849f62b6fb050cfa7a546d6486)

# Methods

**Models:** 
1. Mask RCNN
2. Cellpose

**Techniques:** 
1. Mosaic
2. Add extra data for the `SH-SY5Y` cell line from LIVECell dataset which is the predecessor of this dataset
3. Data Augmentation (Flip left/right, Flip up/down, Crop, Add noise, Rotation
4. L2 Regularization
5. Training Size Model (Cellpose's assistant model)

# Repository structure

`browser`: contains deployment files

`models`: contains Mask RCNN and Cellpose packages

`technique`: contains helper package and some test files

`train-dir`: contains eda, prepare data, training files

# Visualize output

**Mask RCNN**

![Mask RCNN's output](https://www.kaggleusercontent.com/kf/99178893/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..l8Wl85z6nzlrqVAayOnxOw.bO0p5vI2qtaQfsKaI8uJS2zSFf6dp_X9nnBAfhRAVQ3CFRHrTuZyaYfTCncHQ9GD45E_u5s5ngwF_3k0n7Iikkcs8iCcSmwCQ_gD7cVywLwSgrteaHgubmdkjxcdaybu49gNEogHaTa1jguVsWtJ72Z3nP75foKrhFoC3XQ_BZTrZ6kRjgmJ6QjNPMq3KisXU8EAG92yKbOGEpVWBejujxbW89aXWBvThmqEquluHD7ziWuPx7JasAZwXZOfnpZKF49NTGcVqTbTuqFZzojh1_8--9WyiwCdo1OjSOSl6cgz_wISxv_nL-FYnF8rf7qu0uhqNtdIvwJsudsYIce6IpPh5on-57TYRKV9yzibKqSpzVnk6QBPMFDEAK-i07z30lHiXL3i4pYiQUaUVJUFWLqayFFZRuFaYrbv8tEqoPTJyqazfhQUV4y9JlCPQYPS_jXzglW90gpW3TQUCxXhfX45wIJg6YhiuICGt4rMxLdZPVCqMiTtd1b72F9pRJ9D2ydi7mqV-ZR7xRGpPNLJtjHJgMsUtB68ek9KIqqwPFPIbzU33tnCetW7f5_QPS9dhmG-957ijKPcgwAQI7ozikPRFVFV7I8H_MzjxD-k-z0EA03OQpvDoYugwuFrOhguQd2NwM8nCjylROt9YKdvd3vLjAtwES5f9DyzKvKrAho.0GZ_GL4_mZ_G1A8RX5sXPQ/__results___files/__results___33_1.png)

**Cellpose**
![Cellpose's output](https://www.kaggleusercontent.com/kf/99398156/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..r6186lrIdpkEArAZKMt0iw.68O_B2egyTVQl7X8WFSU0obqJk-DqWNeJ-elL6f8CeAVmGyWHZqfI8JCGs8n8Qr6M3nrGFwTeIQCl4y6FYIfcburApwK16u6OXa-ZRZ49rOTrs87dvle8a-4hpKIQIKBVhSUojtx8Ui-joOMOwse4tH3uJz3DrEePFqfdHeqSOkfrESJxPhnHa7OVyNonKO-LzmX0oxFoTHiCNHcg-Ewltv41ydjTsz0cZRK0-dqeupXoA1rvsR7Hh3DYeqIWTQY-kbBpKR6pZhHUHuuU0W16qfMdvKywpdARoga_0HPLfGp23NOSkmmkUYxffkg0R0TLV5ihf-IGTKpW19cV8x0ecTN5pFsbsou29BZT-mn16SWlR47YT7DCFzv8XJXkfvBVS6plfnD7Hw-bEE--VmY-VS_RGXUa8mZ9RVCV0RdrIpjMYXp2JBNh1sqWyNjwLZgGWAxMzxDwQrP3t1gj7s7JMS-_lU_Wbu5kToS-mpdvLLo8u2GdoX5ukn5bsdqHKzUP-Kic-ogElOB6mioIUySUemTHR4Q5gxuraYyF7ffjfT23GWHj2teutd0a7okZPM8dFjLls2m3hIbGtBOqXcSxISVATf--frGFFJuZUENxe82hbeDpY23k780oT80lvYKyg5M8L_n0DqoDowVp5NbJ8ztyGUR2OQnZk90MZDe19k.ruNPUgnrN6Afm5_5jYs6OQ/__results___files/__results___7_0.png)

# Demo

Using `streamlit` framework to demo on website

...

# Reference

Mask RCNN: https://github.com/leekunhee/Mask_RCNN

Cellpose: https://github.com/MouseLand/cellpose

Streamlit: https://streamlit.io/


