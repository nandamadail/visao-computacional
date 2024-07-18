import cv2
import numpy as np

# Caminhos dos arquivos - ajuste conforme necessário
ARQUIVO_VIDEO = 'visao-computacional/rastreio-pessoas/walking.mp4'
ARQUIVO_MODELO = 'visao-computacional/rastreio-pessoas/frozen_inference_graph.pb'
ARQUIVO_CFG = 'visao-computacional/rastreio-pessoas/ssd_mobilenet_v2_coco.pbtxt'


def verificar_opencv():
    # Verifica se o OpenCV está instalado corretamente
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        print("OpenCV não encontrado. Instale o OpenCV com 'pip install opencv-python'.")


def carregar_modelo(ARQUIVO_MODELO, ARQUIVO_CFG):
    '''
    Carrega o modelo de deep learning do TensorFlow para detecção de objetos.
    ARQUIVO_MODELO: Caminho para o arquivo .pb contendo os pesos do modelo.
    ARQUIVO_CFG: Caminho para o arquivo .pbtxt contendo a configuração do modelo.
    Retorna o modelo carregado.
    '''
    modelo = cv2.dnn.readNetFromTensorflow(ARQUIVO_MODELO, ARQUIVO_CFG)
    return modelo


def aplicar_supressao_nao_maxima(caixas, confiancas, limiar_conf, limiar_supr):
    '''
    Aplica a Supressão Não Máxima para reduzir o número de caixas delimitadoras sobrepostas.
    caixas: Lista de caixas delimitadoras.
    confiancas: Lista de confianças de cada caixa.
    limiar_conf: Limiar de confiança para considerar detecções.
    limiar_supr: Limiar de sobreposição para suprimir caixas redundantes.
    Retorna uma lista de caixas após aplicar a supressão.
    '''
    indices = cv2.dnn.NMSBoxes(caixas, confiancas, limiar_conf, limiar_supr)
    return [caixas[i[0]] for i in indices] if len(indices) > 0 else []


def main():
    '''
    Função principal que executa o sistema de contagem de pessoas.
    '''
    verificar_opencv()

    # Inicializa a captura de vídeo
    captura = cv2.VideoCapture(ARQUIVO_VIDEO)
    detector_pessoas = carregar_modelo(ARQUIVO_MODELO, ARQUIVO_CFG)
    pausado = False
    contador_pessoas = 0
    liberado = True

    while True:
        if not pausado:
            ret, frame = captura.read()
            if not ret:
                break

            # Criação do blob a partir do frame e realização da detecção
            blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
            detector_pessoas.setInput(blob)
            deteccoes = detector_pessoas.forward()

            caixas = []
            confiancas = []

            # Extração das caixas delimitadoras e confianças das detecções
            for i in range(deteccoes.shape[2]):
                confianca = deteccoes[0, 0, i, 2]
                if confianca > 0.5:
                    (altura, largura) = frame.shape[:2]
                    caixa = deteccoes[0, 0, i, 3:7] * np.array([largura, altura, largura, altura])
                    (inicioX, inicioY, fimX, fimY) = caixa.astype("int")
                    caixas.append([inicioX, inicioY, fimX - inicioX, fimY - inicioY])
                    confiancas.append(float(confianca))

            # Aplicação da supressão não máxima para finalizar as caixas delimitadoras
            caixas_finais = aplicar_supressao_nao_maxima(caixas, confiancas, limiar_conf=0.5, limiar_supr=0.4)
            numero_pessoas = len(caixas_finais)

            # Contagem de pessoas
            if numero_pessoas > 0 and liberado:
                contador_pessoas += numero_pessoas
                liberado = False
            elif numero_pessoas == 0:
                liberado = True

            # Desenho das caixas e exibição do número de pessoas detectadas
            for (inicioX, inicioY, largura, altura) in caixas_finais:
                cv2.rectangle(frame, (inicioX, inicioY), (inicioX + largura, inicioY + altura), (0, 255, 0), 2)
            cv2.putText(frame, f"Pessoas: {contador_pessoas}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Exibição do frame processado e controle de pausa/play
        cv2.imshow("Sistema de Contagem de Pessoas", frame)
        
        tecla = cv2.waitKey(30) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('p'):
            pausado = not pausado

    # Liberação dos recursos ao finalizar
    captura.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
