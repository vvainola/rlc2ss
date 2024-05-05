<Qucs Schematic 1.1.0>
<Properties>
  <View=206,100,800,502,1.77156,0,0>
  <Grid=10,10,1>
  <DataSet=diode.dat>
  <DataDisplay=diode.dpl>
  <OpenDisplay=1>
  <Script=diode.m>
  <RunScript=0>
  <showFrame=0>
  <FrameText0=Title>
  <FrameText1=Drawn By:>
  <FrameText2=Date:>
  <FrameText3=Revision:>
</Properties>
<Symbol>
</Symbol>
<Components>
  <L L1 1 580 220 -26 10 0 0 "1 mH" 1 "" 0>
  <R R3 1 490 220 -26 15 0 0 "1 kOhm" 1 "26.85" 0 "0.0" 0 "0.0" 0 "26.85" 0 "european" 0>
  <Vdc V1 1 240 300 18 -26 0 1 "1 V" 1>
  <S4Q_S S1 1 280 220 -26 19 0 0 "" 1 "" 0 "" 0 "" 0 "" 0>
  <R R1 1 340 220 -26 15 0 0 "1 kOhm" 1 "26.85" 0 "0.0" 0 "0.0" 0 "26.85" 0 "european" 0>
  <DIODE_SPICE D1 1 410 220 -26 15 0 0 "" 1 "" 0 "" 0 "" 0 "" 0>
  <GND * 1 240 410 0 0 0 0>
  <DIODE_SPICE D2 1 450 320 15 -26 0 1 "" 1 "" 0 "" 0 "" 0 "" 0>
  <R R2 1 450 380 15 -26 0 1 "1 kOhm" 1 "26.85" 0 "0.0" 0 "0.0" 0 "26.85" 0 "european" 0>
</Components>
<Wires>
  <520 220 550 220 "" 0 0 0 "">
  <450 220 460 220 "" 0 0 0 "">
  <240 220 250 220 "" 0 0 0 "">
  <240 220 240 270 "" 0 0 0 "">
  <370 220 380 220 "" 0 0 0 "">
  <440 220 450 220 "" 0 0 0 "">
  <240 330 240 410 "" 0 0 0 "">
  <450 220 450 290 "" 0 0 0 "">
  <610 220 610 350 "" 0 0 0 "">
  <450 350 610 350 "" 0 0 0 "">
  <240 410 450 410 "" 0 0 0 "">
  <370 220 370 220 "N_D1_pos" 370 120 0 "">
  <450 350 450 350 "N_D2_pos" 359 330 0 "">
  <450 220 450 220 "N_D_neg" 450 150 0 "">
</Wires>
<Diagrams>
</Diagrams>
<Paintings>
</Paintings>
