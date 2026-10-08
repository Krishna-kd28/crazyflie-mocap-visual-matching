# Pipeline algorithms and meaning

The pipeline synchronizes recordings and mocap, corrects camera poses using
shared room features, then fits the simulated camera response. Every image uses
the native 320 x 240 Crazyflie geometry and recorded real-image intensities.

## 1. Time alignment and mocap interpolation

Fit an affine map from the camera's deck clock to its host-arrival clock using
iteratively reweighted least squares:

$$\hat t_h=t_{h,0}+a(t_d-t_{d,0})+b,\qquad t_q=\hat t_h+\Delta.$$

Here $t_d$ is device time and $t_{h,0}$ is the first host receipt time. $a$
accounts for clock drift, $b$ is the centered intercept, and $\Delta$ is the
remaining image-to-mocap delay. All terms use seconds. A positive $\Delta$ queries
a later mocap pose. Host receipt time includes transport latency.

The delay search tests 301 values from -0.75 to +0.75 seconds in 5 ms steps.
Track image corners forward and backward, reject inconsistent or saturated
tracks, and invert the lens to obtain camera rays. For each candidate delay,
form the essential matrix from the relative mocap-derived camera motion and
score the Sampson residual of the same tracks. Use medians per pair and across
pairs so numerous tracks in one image do not dominate. Alternating 10-second
blocks fit and check the delay. A boundary optimum or no held-out improvement
falls back to zero lag and is recorded explicitly.

Positions use linear interpolation; rotations use quaternion SLERP. Reject
queries outside mocap support, gaps above 50 ms, camera-arrival outliers above
50 ms, and intervals near large orientation jumps. Missing poses remain invalid.

The configured camera-to-body extrinsic gives

$$R_{MC}=R_{MB}R_{BC},\qquad p_{MC}=p_{MB}+R_{MB}p_{BC}.$$

$M$ is mocap, $B$ is the tracked body, $C$ is the optical camera (right, down,
forward). The supplied lever arm is $(0.027,0,-0.007)$ m in the calibrated body
frame. Recalibrate it after recreating the rigid body or changing its pivot.

Code: `baseline.py`, `check_motion_sync.py`, `prepare_recordings.py`.

## 2. Camera lens and reconstruction coordinates

Keep raw frames unchanged. Convert only feature coordinates through the active
inverse lens model when performing geometry. The included full-field model is

$$r_d=\theta(1+k_1\theta^2+k_2\theta^4),\qquad\theta=\arctan r,$$

with $k_1=0.2650426865$, $k_2=-0.0206977526$, and the supplied intrinsics
$f_x=181.153324$, $f_y=182.146828$, $c_{0x}=166.454976$,
$c_{0y}=75.996424$ pixels. These coefficients belong to the angle-based model in
`lens_model.json`. The camera file also records a radial-tangential calibration;
its coefficients describe a different projection function.

For rendering, gsplat's 3DGUT fisheye path directly produces the raw image
geometry. The fallback renders an expanded pinhole canvas and samples it using
the inverse target-pixel mapping. Its expanded field of view covers the rays needed at the output corners.

The room registration is a similarity:

$$p_G=\lambda R_{GM}p_M+t_G,\qquad R_{GC}=R_{GM}R_{MC}.$$

$G$ is the nominal metric scene; its original conversion is 0.85 metres per
scene unit. `dataparser_transforms.json` and `world_frame.json` then map this
frame into checkpoint coordinates $S$. Render with $T_{CS}=T_{SC}^{-1}$. NPZ files state which transform
is stored; `T_GC` has translation in metres.

Code: `fixed_features_common.py`, `prepare_fixed_feature_frames.py`,
`render_baseline.py`.

## 3. Bootstrap the room and camera-axis correction

Render a 64-view atlas at the packaged reference poses. Match real keyframes to
atlas images using SuperPoint and MINIMA LightGlue. Expected rendered depth
lifts matches into the scene; alpha, depth range and depth spread reject weak
samples. RANSAC PnP supplies provisional visual camera poses.

Fit a shared scene rotation/translation/scale and a right-multiplied camera
rotation against those visual poses. Search four broad initial yaws and use
robust least squares. Re-estimate image/mocap timing with the camera correction,
then refit and apply the shared correction. The configured lever arm remains
fixed during this rotation calibration.

The atlas specifies camera poses for rendering the configured scene. Supply
a matching atlas and seed registration for each reconstruction.

Code: `bootstrap_registration.py`.

## 4. Build persistent fixed-room features

1. Render at the bootstrapped poses and match each real/render pair.
2. Match renders 1, 2 and 5 frames apart with stock LightGlue; chain consistent
   keypoints into tracks.
3. Intersect rays from known render poses to recover a scene point $X_j$:

$$X_j=\arg\min_X\sum_k\|(I-d_kd_k^T)(X-p_k)\|^2.$$

$p_k$ is the rendering camera centre, $d_k$ its unit ray for the observed pixel,
and $I$ is the 3 x 3 identity matrix. All are expressed in $G$.

4. Require at least three observations, 0.3 seconds of time span, 3 degrees of
   parallax, render reprojection error at most 1.5 px, real median reprojection
   at most 12 px and real/render triangulations within 0.25 m. Reject configured
   floor/central-gate regions; these exclusions are scene-specific.
5. Merge duplicates within 5 cm and choose spatially separated candidates that
   improve frame coverage, targeting at least five features per frame, up to
   250 candidates. CoTracker3 can propagate features, with a two-direction
   agreement check. Guided gradient-patch NCC extends remaining observations.

These checks reduce moved-object associations; they cannot prove that an object
is permanently fixed. The separate review interface stores explicit candidate
decisions. Confidence and tracking-distance uncertainty remain in the dataset.

Code: `auto_features.py`, `auto_feature_viewer.py`, `track_features_batch.py`,
`track_fixed_features.py`, `extend_fixed_feature_observations.py`.

## 5. Global brute-force registration

Starting from $R_i,p_i$, apply the seven-parameter correction
$\vartheta=(\omega_x,\omega_y,\omega_z,\delta_x,\delta_y,\delta_z,s)$:

$$R_i'=\exp([\omega]_\times)R_i,\qquad
p_i'=e^s\exp([\omega]_\times)(p_i-c)+c+\delta.$$

$c$ is the mean starting camera centre, used as a convenient rotation/scale
pivot. $s$ is log scale, so $e^s$ multiplies distances. For scene point $X_j$,
project $R_i'^T(X_j-p_i')$ with the pinhole intrinsics to obtain $\hat u_{ij}$.
The measured raw point is inverse-lens-corrected to the same coordinate system.

$$L(\vartheta)=\frac1W\sum_{ij}w_{ij}
\rho\!\left(\frac{\|\hat u_{ij}-u_{ij}\|}{\sigma_{ij}}\right)
+\frac1{2W}\sum_k(\vartheta_k/\tau_k)^2,$$

$$\rho(e)=\begin{cases}e^2/2,&e\le1\\e-1/2,&e>1.\end{cases}$$

$W=\sum w_{ij}$, $w_{ij}$ divides each seed's influence among its tracked
observations, $\sigma_{ij}$ is pixel uncertainty and $\tau_k$ is the allowed
correction scale for the weak prior. Points behind the camera receive a fixed
penalty. The loss uses undistorted pinhole pixels; raw images remain unchanged.

The first grid has nine samples on each of six pose axes ($9^6=531441$), scale
fixed. Subsequent grids use seven samples per pose axis and three log-scale
offsets around retained candidates. Six levels keep 6, 4, 3, 2, 2 and 1 diverse
centres, followed by Powell minimization of the same objective. Widths are
listed in the code, and every evaluated level is recorded in `search.json`.
The wrapper compares free/fixed-scale fits on a deterministic sample of up to
120 observed frames and chooses by withheld-frame RMS.

Code: `brute_force_registration.py`, `run_alignment.py`.

## 6. Correct each frame independently

For each frame with observations, search six parameters:

$$R_i'=\exp([\omega_i]_\times)R_i,\qquad p_i'=p_i+\delta_i.$$

Use that frame's observations and the same robust/prior structure. Five levels
each contain $7^6$ candidates; start at +/-4 degrees and +/-0.25 m, shrinking
each width by 0.35 per level. Then run Powell refinement. This is 588245 grid
candidates per fitted frame before the local refinement. No temporal smoothing
is enabled by default.

Frames without observations may receive interpolated/nearby corrections within
the configured three-second gap limit; otherwise retain the global pose. The
`mode` field distinguishes `fitted`, `interpolated`, and `global`. Having few
observations can yield a numerically fitted but weakly constrained pose.

Code: `per_frame_registration.py`.

## 7. Fit simulated camera appearance

Freeze all poses and geometry. Convert RGB render $I$ to grayscale
$g=0.299I_R+0.587I_G+0.114I_B$, with intensities in [0,1]. The selected response is

$$\hat I(p)=\operatorname{clip}_{[0,1]}\left[
b+a(e_rh_r)^\beta\,\bar g(p)^\gamma
\exp\{-v_1q(p)-v_2q(p)^2+\ell^TB(d_p)+c^TF_r\}\right].$$

| Term | Meaning |
|---|---|
| $\bar g$ | Grayscale render with selected Gaussian blur; current sigma 0.8 px |
| $a,b,\gamma$ | Shared gain, bounded black offset and tone exponent |
| $e_r$ | Reported `(exposure_ms * digital_gain) / (8.33 * 1.5)` |
| $h_r$ | Small fitted gain for a known run; use 1 for an uncalibrated new run |
| $\beta$ | Empirical exposure-response exponent |
| $q(p)$ | `((u-c0x)/200)^2 + ((v-c0y)/200)^2` |
| $v_1,v_2$ | Radial response coefficients |
| $d_p$ | Unit camera ray rotated into the metric scene |
| $B(d)$ | Eight directional terms: $x,y,z,xy,xz,yz,x^2-y^2,3z^2-1$ |
| $\ell$ | Coefficients of per-ray directional brightness |
| $F_r,c$ | Central-ray directional terms and simulated bright-content summaries, with fitted coefficients |

Fit bounded parameters by robust least squares using stable-feature pixels,
spatially uniform 40 x 40 tile means, radial-region means and a dark quantile.
Runs have equal influence. Test candidate forms and blur settings on development
frames, freeze the selection, then score reserved frames. The 40-pixel tiles distribute the loss across the image while a shared
response supplies the correction.

The temporal protocol uses 10-second blocks with one-second guards: three
blocks train, one selects and one checks. Response extensions preserve the
training blocks and reserve additional guard-gap windows before fitting, as
specified in [the workflow guide](WORKFLOW.md). Appearance validation measures
same-recording agreement conditional on the fitted camera poses. The fitted
response approximates the combined effects of scene lighting and camera response;
validate transfer to a new scene or camera separately.

At inference, the transform uses simulated RGB, simulated pose and exposure/gain
metadata only. Saved JSON contains all
coefficients, including known-run gains.

Code: `fit_exposure.py`, `fit_appearance.py`,
`fit_camera_response.py`, `export_appearance.py`.

## 8. Metrics and how to interpret them

For pixel residual $d_p=\hat I_p-I_p$ on a declared support $\Omega$:

$$\mathrm{MAE}=\frac1{|\Omega|}\sum_{p\in\Omega}|d_p|,\qquad
\mathrm{RMSE}=\sqrt{\frac1{|\Omega|}\sum_{p\in\Omega}d_p^2}.$$

RMSE is the square root of MSE and penalizes large errors more. Signed bias is
$\operatorname{mean}(d_p)$; also report per-frame absolute bias so opposite
bright/dark errors do not cancel. Intensities are fixed in [0,1], not rescaled
independently per image.

SSIM compares local means, variances and covariance using an 11 x 11 Gaussian
window, sigma 1.5, with constants $0.01^2$ and $0.03^2$:

$$\mathrm{SSIM}=
\frac{(2\mu_I\mu_J+C_1)(2\operatorname{cov}(I,J)+C_2)}
{(\mu_I^2+\mu_J^2+C_1)(\operatorname{var}I+\operatorname{var}J+C_2)}.$$

Edge precision/recall count Canny edges within two pixels of the other image's
edges; $F_1=2PR/(P+R)$. Canny thresholds are 15/35 on 8-bit intensities after
sigma-0.8 analysis blur. Full-image metrics exclude a five-pixel outer border.
Stable-feature support and full-image support are reported separately.

The initial display score combines complementary components:

$$Q=100\left[e^{-\mathrm{MAE}/0.10}\,
\frac{1+\mathrm{SSIM}}2\,F_1\right]^{1/3}.$$

It is an engineering score, not a perceptual probability. The final appearance
selection uses a different declared loss to discourage overly dark/blurry fits:

$$J=\mathrm{MAE}_{full}+0.5\mathrm{MAE}_{support}
+0.5B_{radial}+0.05(1-\mathrm{SSIM}_{full})
+0.10(1-F_{1,full})+0.05(1-F_{1,support}),$$

where $B_{radial}$ is the mean absolute signed bias across radial regions, in
[0,1] intensity units. Report each component as well as the combined number.

Pose reprojection RMS/median are distances between projected scene features and
measured features, in pinhole pixels. They measure feature agreement, not the
metric accuracy of the full camera pose. Patch verification renders the fitted
pose and searches for small gradient-patch shifts as an additional visual check.

Code: `image_similarity.py`, `appearance_metrics.py`,
`verify_fixed_feature_registration.py`.
