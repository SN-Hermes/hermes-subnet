# SN SubQuery Hermes incentive mechanisms

The `SN SubQuery Hermes` incentive system consists of two primary components: the FactorScore alignment between miners and validators in synthetic challenges, and the response time weighting mechanism. The final score is derived from the comprehensive evaluation of these two factors.



**Factual Accuracy Score**

 Validators randomly generate numerical project-related challenge questions and send them to miners. The validator simultaneously produces a standard answer using its own Agent, then collects responses from miners and evaluates them using FactorScore. The scoring range spans from 0 to 10.



**Elapsed Time Weight**

Upon receiving miner responses, validators use their own standard answer generation time as the baseline reference. The weight decreases quadratically based on response time deviation from this benchmark.

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?w%20=%20\frac{1}{1%20+%20\left(\frac{elapsed\_time}{ground\_truth\_cost}\right)^2}" />
</p>



**Final Score Calculation**

$$
\text{Final Score} = \text{FactorScore} \times \text{Elapsed Time Weight}
$$
