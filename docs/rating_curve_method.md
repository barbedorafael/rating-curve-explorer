# Curva‑chave (Stage–Discharge)

## Modelo
Q = a * (H − h0)^b

## Segmentação
- Quebras em nível (`level_breaks`): `[minH, b1]`, `(b1, b2]`, …, `(b_n, maxH]`.

## Ajuste
- Preferência: **least-squares não‑linear** (SciPy `least_squares` com loss Huber).
- Restrições: `a > 0`, `1.2 ≤ b ≤ 6`, `h0 < min(H)`.
- Retornamos `RMSE` por segmento para diagnóstico.

## Boas práticas
- Garantir `Q > 0` e remover outliers evidentes.
- Verificar coerência física do `h0` (abaixo do regime observado).
- Guardar parâmetros por estação/segmento para reuso.