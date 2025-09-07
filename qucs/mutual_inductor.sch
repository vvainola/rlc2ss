<Qucs Schematic 1.1.0>
<Properties>
  <View=0,-60,935,800,1.21,0,97>
  <Grid=10,10,1>
  <DataSet=mutual_inductor.dat>
  <DataDisplay=mutual_inductor.dpl>
  <OpenDisplay=1>
  <Script=mutual_inductor.m>
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
  <GND * 1 230 390 0 0 0 0>
  <Vdc V1 1 330 290 -26 18 0 0 "1 V" 1>
  <Vdc V2 1 330 380 -26 18 0 0 "1 V" 1>
  <Vdc V3 1 330 480 -26 18 0 0 "1 V" 1>
  <L L1 1 580 290 -26 10 0 0 "1 H" 1 "" 0>
  <L L2 1 580 380 -26 10 0 0 "1 H" 1 "" 0>
  <L L3 1 580 480 -26 10 0 0 "1 H" 1 "" 0>
  <K_SPICE K12 1 860 290 -26 17 0 0 "L1" 1 "L2" 1 "0.9" 1>
  <K_SPICE K21 1 860 400 -26 17 0 0 "L2" 1 "L3" 1 "0.9" 1>
  <K_SPICE K31 1 860 510 -26 17 0 0 "L3" 1 "L1" 1 "0.9" 1>
  <R R1 1 450 290 -26 15 0 0 "10 Ohm" 1 "26.85" 0 "0.0" 0 "0.0" 0 "26.85" 0 "european" 0>
  <R R2 1 450 380 -26 15 0 0 "10 Ohm" 1 "26.85" 0 "0.0" 0 "0.0" 0 "26.85" 0 "european" 0>
  <R R3 1 450 480 -26 15 0 0 "10 Ohm" 1 "26.85" 0 "0.0" 0 "0.0" 0 "26.85" 0 "european" 0>
  <IProbe Pr1 1 430 580 -26 -35 0 2>
</Components>
<Wires>
  <300 290 300 380 "" 0 0 0 "">
  <230 380 230 390 "" 0 0 0 "">
  <230 380 300 380 "" 0 0 0 "">
  <300 380 300 480 "" 0 0 0 "">
  <360 290 420 290 "" 0 0 0 "">
  <360 380 420 380 "" 0 0 0 "">
  <360 480 420 480 "" 0 0 0 "">
  <480 290 550 290 "" 0 0 0 "">
  <480 380 550 380 "" 0 0 0 "">
  <480 480 550 480 "" 0 0 0 "">
  <610 290 610 380 "" 0 0 0 "">
  <610 380 610 480 "" 0 0 0 "">
  <460 580 610 580 "" 0 0 0 "">
  <610 480 610 580 "" 0 0 0 "">
  <300 480 300 580 "" 0 0 0 "">
  <300 580 400 580 "" 0 0 0 "">
</Wires>
<Diagrams>
</Diagrams>
<Paintings>
</Paintings>
