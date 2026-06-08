# Speculative Decoding Algorithm Specification

## Overview
This document outlines the Speculative Decoding algorithm designed to accelerate inference from large autoregressive models (like Transformers). The core idea is to use a smaller, faster approximation model to generate multiple speculative tokens sequentially, and then use the large target model to evaluate all these speculative tokens in parallel in a single forward pass.

## Definitions
* **Target Model ($M_p$)**: The large, slow model whose inference we want to accelerate.
* **Approximation Model ($M_q$)**: A smaller, faster model used to generate speculative prefixes.
* **$\gamma$**: The number of speculative tokens to generate per iteration.
* **prefix**: The current sequence of tokens.

## Prerequisites: Standardized Sampling
All sampling methods (argmax, top-k, nucleus, temperature) must be cast into standard sampling from an adjusted probability distribution.
* For example, argmax sampling is equivalent to zeroing out non-max elements of the probability distribution and normalizing it to 1.
* Both $M_p$ and $M_q$ must use the *same* probability standardization.

## Algorithm Steps

**Input:** $M_p$, $M_q$, `prefix`, $\gamma$

### 1. Speculative Generation (Sequential)
Generate $\gamma$ guesses autoregressively using the approximation model $M_q$.
* For $i = 1$ to $\gamma$:
    * Calculate distribution: $q_i(x) = M_q(\text{prefix} + [x_1, ..., x_{i-1}])$
    * Sample token: $x_i \sim q_i(x)$

### 2. Parallel Evaluation
Run the target model $M_p$ in parallel to get the distributions for the original prefix and all speculative steps.
* Calculate distributions:
    $p_1(x), ..., p_{\gamma+1}(x) = M_p(\text{prefix}), ..., M_p(\text{prefix} + [x_1, ..., x_\gamma])$

*(Implementation Note: This is usually done in a single forward pass of $M_p$ by passing the sequence `prefix + [x_1, ..., x_\gamma]` and extracting the probability distributions for the last $\gamma + 1$ positions).*

### 3. Acceptance / Rejection Phase
Determine how many speculative tokens $n$ are accepted.
* Draw $\gamma$ random numbers from a uniform distribution: $r_1, ..., r_\gamma \sim U(0,1)$
* Set $n = \gamma$
* For $i = 1$ to $\gamma$:
    * If $r_i > \frac{p_i(x_i)}{q_i(x_i)}$:
        * $n = i - 1$
        * Break out of the loop (reject this token and all subsequent ones).

### 4. Distribution Adjustment & Resampling
Sample the final token to add to the sequence, fixing the first rejected token or adding an additional one if all were accepted.
* Initialize the distribution for the next token: $p'(x) = p_{n+1}(x)$
* If $n < \gamma$ (meaning a token was rejected):
    * Adjust the distribution to account for the rejection:
        $p'(x) = \text{norm}(\max(0, p_{n+1}(x) - q_{n+1}(x)))$
        *(Implementation Note: `norm` means dividing every element by the sum of the array so the probabilities sum to 1).*
* Sample the final token $t \sim p'(x)$.

### 5. Update Sequence
* Return the new sequence: `prefix + [x_1, ..., x_n, t]`
* *(The algorithm successfully produced $n + 1$ new tokens in this single iteration).*