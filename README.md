# MotorAI

MotorAI é um modelo de linguagem experimental e altamente otimizado, construído do zero sobre uma arquitetura de Spiking Neural Network (SNN). 

O projeto foi desenvolvido para demonstrar que o processamento de linguagem natural e a inferência de alto rendimento (throughput) podem ser alcançados sem a necessidade de datacenters ou quantidades massivas de VRAM. O foco principal é a obtenção de cognição semântica utilizando hardware de borda (edge computing) e processamento físico/termodinâmico de tensores.

> Status do Projeto: Em Treinamento Ativo (Work in Progress)
> O modelo encontra-se em fase de pré-treinamento e Supervised Fine-Tuning (SFT). A arquitetura atual já demonstra domínio sobre a sintaxe da língua portuguesa e estabilidade em gatilhos de parada (EOS). O desenvolvimento atual foca no escalonamento da rede de 1.4M para 5.5M de parâmetros, visando expandir a entropia necessária para a compreensão semântica profunda.

## Arquitetura Base

O paradigma tradicional de Transformers densos foi substituído por uma malha de processamento temporal baseada em redes neurais pulsantes:

- MotorCore: Matriz responsável pela propagação de energia cinética e processamento espacial, utilizando operadores Laplacianos dilatados.
- ALIFCell: Neurônios do tipo Adaptive Leaky Integrate-and-Fire, que disparam pulsos baseados em limiares dinâmicos de voltagem (threshold), simulando o comportamento biológico.
- Precisão Matemática: Fluxo de tensores estritamente em FP32 via PyTorch CUDA.
- Tokenização: ByteLevel BPE customizado e otimizado para o idioma português.

## Benchmarks de Hardware (V3 Toddler - 1.39M Parâmetros)

A engenharia SNN do MotorAI é direcionada à otimização extrema para hardwares legados. Os testes abaixo foram homologados localmente utilizando processadores convencionais e uma NVIDIA GTX 1050 Ti.

| Métrica | Desempenho Medido |
| :--- | :--- |
| Throughput CPU | ~189 Tokens/segundo |
| Throughput GPU | ~135 Tokens/segundo |
| Latência Média | 3.20 a 5.83 ms por token |
| Consumo de VRAM | 30 MB |
| Consumo de RAM | ~1016 MB |
| Tamanho do Arquivo | 16.03 MB (.pth) |

*Nota Técnica: Em arquiteturas com contagem de parâmetros extremamente reduzida (como o modelo de 1.39M), a inferência em CPU supera a GPU em throughput devido à ausência de latência de transferência no barramento PCIe. Os pesos do modelo cabem quase integralmente nos caches L2/L3 do processador.*

## Roadmap de Desenvolvimento

- [x] Implementação da malha temporal FP32 (MotorCore + ALIF).
- [x] Pré-treinamento sintático (Corpus Wikipedia PT-BR + OpenSubtitles).
- [x] Estabilização do gatilho de parada (EOS Token) e eliminação de loops infinitos.
- [ ] Recuperação de SFT e aplicação de balanceamento de dados de instrução.
- [ ] Escalonamento da arquitetura para a versão V5 Child (5.5M Parâmetros e Vocabulário 4k).
- [ ] Implementação de inferência bare-metal e deploy local via API.

## Instruções de Uso (Inferência Local)

Para clonar o repositório e executar o script de inferência básica utilizando o checkpoint atual:

```bash
git clone [https://github.com/LuanOlympio2/motorai.git](https://github.com/LuanOlympio2/motorai.git)
cd motorai
python motor_v4_toddler.py
