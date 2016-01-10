/**
 * ------------------------------------------------------------
 * Copyright (c) SILAB , Physics Institute of Bonn University 
 * ------------------------------------------------------------
 */

`timescale 1ps / 1ps

`include "firmware/src/fe65p2_mio.v"
 
module tb (
    input wire FCLK_IN, 

    //full speed 
    inout wire [7:0] BUS_DATA,
    input wire [15:0] ADD,
    input wire RD_B,
    input wire WR_B,
    
    //high speed
    inout wire [7:0] FD,
    input wire FREAD,
    input wire FSTROBE,
    input wire FMODE,
    
    output wire CLK_BX,
    input wire [64*64-1:0] HIT,
    input wire TRIGGER,
    output wire [1:0] RESET
);   


wire [19:0] SRAM_A;
wire [15:0] SRAM_IO;
wire SRAM_BHE_B;
wire SRAM_BLE_B;
wire SRAM_CE1_B;
wire SRAM_OE_B;
wire SRAM_WE_B;

//wire [1:0] RESET;
//wire CLK_BX;
wire TRIGGER_DUT;
wire CLK_CNFG;
wire EN_PIX_SR_CNFG;
wire LD_CNFG;
wire SI_CNFG; 
wire SO_CNFG;
wire PIX_D_CONF;
wire CLK_DATA; 
wire OUT_DATA;
wire HIT_OR;
wire INJ;

fe65p2_mio fpga (
    .FCLK_IN(FCLK_IN),
        
    .BUS_DATA(BUS_DATA), 
    .ADD(ADD), 
    .RD_B(RD_B), 
    .WR_B(WR_B), 
    .FDATA(FD), 
    .FREAD(FREAD), 
    .FSTROBE(FSTROBE), 
    .FMODE(FMODE),

    .SRAM_A(SRAM_A), 
    .SRAM_IO(SRAM_IO), 
    .SRAM_BHE_B(SRAM_BHE_B), 
    .SRAM_BLE_B(SRAM_BLE_B), 
    .SRAM_CE1_B(SRAM_CE1_B), 
    .SRAM_OE_B(SRAM_OE_B), 
    .SRAM_WE_B(SRAM_WE_B), 
        
    .DUT_RESET(RESET) ,
    .DUT_CLK_BX(CLK_BX), 
    .DUT_TRIGGER(TRIGGER_DUT),
    .DUT_INJ(INJ),
    .DUT_CLK_CNFG(CLK_CNFG), 
    .DUT_EN_PIX_SR_CNFG(EN_PIX_SR_CNFG), 
    .DUT_LD_CNFG(LD_CNFG), 
    .DUT_SI_CNFG(SI_CNFG), 
    .DUT_SO_CNFG(SO_CNFG),
    .DUT_PIX_D_CONF(PIX_D_CONF),
    .DUT_CLK_DATA(CLK_DATA), 
    .DUT_OUT_DATA(OUT_DATA),
    .DUT_HIT_OR(HIT_OR)
);   

wire [64*64-1:0] ANA_HIT;

wire TRIGGER_FE;
assign  TRIGGER_FE = TRIGGER_DUT | TRIGGER;

wire PrmpVbp, vthin1, vthin2, vff, VctrCF0, VctrCF1, PrmpVbnFol, vbnLcc, compVbn, preCompVbn, RefVbn;
wire [31:0] test_out1L, test_out1R, test_out2L, test_out2R, test_out2bL, test_out2bR;
wire [63:0] outDiscInvR;
wire HIT_OR_N, OUT_DATA_N;
wire OUT_DATA_P;

fe65p2 dut(                   
    ANA_HIT, 
    RESET ,
    CLK_BX, 
    TRIGGER_FE,
    HIT_OR, 
    HIT_OR_N,
    CLK_CNFG,
    EN_PIX_SR_CNFG, 
    LD_CNFG, 
    SI_CNFG, 
    SO_CNFG,
    PIX_D_CONF,
    CLK_DATA, 
    OUT_DATA_P,
    OUT_DATA_N,
    ~INJ, 
    PrmpVbp, vthin1, vthin2, vff, VctrCF0, VctrCF1, PrmpVbnFol, vbnLcc, compVbn, preCompVbn, RefVbn,
    test_out1L, test_out1R, test_out2L, test_out2R, test_out2bL, test_out2bR, outDiscInvR
);

assign #5500 OUT_DATA = OUT_DATA_P;

//SRAM Model
reg [15:0] sram [1048576-1:0];
assign SRAM_IO = !SRAM_OE_B ? sram[SRAM_A] : 16'hzzzz;
always@(negedge SRAM_WE_B)
    sram[SRAM_A] <= SRAM_IO;

assign ANA_HIT = HIT;

initial begin
    
    $dumpfile("fe65p2.vcd");
    $dumpvars(0);
    
    //force dut.i_output_data.iser_div.cnt = 4'b0;
    //#10 force CLK_DATA = 4'b0;
    //#100000 force CLK_DATA = 4'b1;
    //#10000 release CLK_DATA;
    //#50000 release dut.i_output_data.iser_div.cnt;

end

endmodule
