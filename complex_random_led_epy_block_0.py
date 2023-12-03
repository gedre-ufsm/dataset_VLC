import numpy as np
from gnuradio import gr
import pmt

class blk(gr.sync_block):
    """Embedded Python Block example - encuentra el máximo en un vector de longitud 512"""

    def __init__(self, vectorSize=1024):  # no se necesitan parámetros adicionales
        """Inicializa el bloque"""
        gr.sync_block.__init__(
            self,
            name='Encuentra el Máximo',  # Aparecerá en GRC
            in_sig=[(np.float32,vectorSize)],  # Tipo de datos de entrada: float
            out_sig=[(np.float32,1)]  #Salida será el índice del máximo valor en el vector
        )
        
        self.prev_delay = 25
        self.delay = 25
        self.portName = 'dlyOutput'
        self.message_port_register_out(pmt.intern(self.portName))
        
  
    def work(self, input_items, output_items):
        """Encuentra el índice del máximo en el vector de entrada"""
        # Usamos np.argmax para encontrar el índice del máximo valor en el vector
        for vectorIndex in range(len(input_items[0])):
        	max_index = np.argmax(input_items[0][vectorIndex])
        	# Colocamos el índice del máximo en la salida
        	output_items[0][vectorIndex] = max_index
        
        self.delay = max_index
        if (self.delay != self.prev_delay):
                PMT_msg = pmt.from_long(self.delay)
                self.message_port_pub(pmt.intern(self.portName), PMT_msg)
                self.prev_delay = self.delay	
        	
        		
        return len(output_items[0])
