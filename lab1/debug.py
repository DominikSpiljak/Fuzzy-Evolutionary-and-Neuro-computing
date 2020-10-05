class Debug:

    @staticmethod
    def debug_print(domain, heading_text=""):
        print(heading_text)
        for element in domain:
            print(element)
        print("Kardinalitet domene je: {}".format(domain.getCardinality()))
