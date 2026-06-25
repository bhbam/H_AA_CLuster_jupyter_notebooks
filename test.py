Taumcoll = (
            hist.Hist.new
            .StrCat(["Taumcoll"], name="hmassjet")  #
            .Reg(50, 0, 250, name="massjet", label="Mass[GeV]")
            .Int64()
        )
Taumcoll.fill(hmassjet="Taumcoll", massjet=taumcoll)  #
