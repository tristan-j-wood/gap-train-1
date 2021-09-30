import xml.etree.cElementTree as ET
from xml.dom import minidom


class PMID:
    """Path integral molecular dynamics class to run PMID simulations using
       the i-PI Python interface."""

    def write_xml_file(self):
        """Writes the xml input file required for i-PI"""

        root = ET.Element("simulation", verbosity='high')

        return NotImplementedError

    def run_pmid(self):

        return NotImplementedError

    def __int__(self):

        self.test = "test"
