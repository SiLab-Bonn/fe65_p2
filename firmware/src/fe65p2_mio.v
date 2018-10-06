/**
 * ------------------------------------------------------------
 * Copyright (c) SILAB , Physics Institute of Bonn University 
 * ------------------------------------------------------------
 */

`timescale 1ps / 1ps

`include "clk_gen.v"

//BASIL includes
`include "utils/bus_to_ip.v"
`include "gpio/gpio.v"

`include "spi/spi.v"
`include "spi/spi_core.v"
`include "spi/blk_mem_gen_8_to_1_2k.v"
   
`include "sram_fifo/sram_fifo_core.v"
`include "sram_fifo/sram_fifo.v"

`include "fei4_rx/fei4_rx_core.v"
`include "fei4_rx/receiver_logic.v"
`include "fei4_rx/sync_master.v"
`include "fei4_rx/rec_sync.v"
`include "fei4_rx/decode_8b10b.v"
`include "fei4_rx/fei4_rx.v"
`include "utils/flag_domain_crossing.v"

`include "utils/cdc_syncfifo.v"
`include "utils/generic_fifo.v"
`include "utils/cdc_pulse_sync.v"
`include "utils/CG_MOD_pos.v"
`include "utils/clock_divider.v"
`include "utils/fx2_to_bus.v"
`include "utils/reset_gen.v"

`include "pulse_gen/pulse_gen.v"
`include "pulse_gen/pulse_gen_core.v"

`include "tdc_s3/tdc_s3.v"
`include "tdc_s3/tdc_s3_core.v"

`include "tlu/tlu_controller.v"
`include "tlu/tlu_controller_core.v"
`include "tlu/tlu_controller_fsm.v"

`include "utils/3_stage_synchronizer.v"
`include "rrp_arbiter/rrp_arbiter.v"
`include "utils/ddr_des.v"

`ifdef COCOTB_SIM //for simulation
    `include "utils/IDDR_sim.v" 
    `include "utils/ODDR_sim.v" 
    `include "utils/BUFG_sim.v" 
    `include "utils/DCM_sim.v" 
    `include "utils/clock_multiplier.v"
    `include "utils/RAMB16_S1_S9_sim.v"
`else
    `include "utils/IDDR_s3.v"
    `include "utils/ODDR_s3.v"
`endif

module fe65p2_mio (
    input wire FCLK_IN, // 48MHz

    //full speed 
    inout wire [15:0] BUS_DATA,
    input wire [15:0] ADD,
    input wire RD_B,
    input wire WR_B,

    //high speed
    inout wire [7:0] FDATA,
    input wire FREAD,
    input wire FSTROBE,
    input wire FMODE,
    
    
    //SRAM
    output wire [19:0] SRAM_A,
    inout wire [15:0] SRAM_IO,
    output wire SRAM_BHE_B,
    output wire SRAM_BLE_B,
    output wire SRAM_CE1_B,
    output wire SRAM_OE_B,
    output wire SRAM_WE_B,
    
    input wire [2:0] LEMO_RX,
    output wire [2:0] LEMO_TX,
    
    inout wire SDA,
    inout wire SCL,
    
    output wire [4:0] LED,
    
    output wire [1:0] DUT_RESET,
    output wire DUT_CLK_BX, 
    
    output wire DUT_TRIGGER,
    output wire DUT_INJ,
    
    output wire DUT_CLK_CNFG, 
    output wire DUT_EN_PIX_SR_CNFG, 
    output wire DUT_LD_CNFG,
    output wire DUT_SI_CNFG,
    
    input wire DUT_SO_CNFG,
    input wire DUT_HIT_OR,
    
    output wire DUT_PIX_D_CONF,  
    
    output wire DUT_CLK_DATA, 
    input wire DUT_OUT_DATA
	 
    
    
);   
    //wire HIT_OR_2;
    //assign HIT_OR_2 = DUT_HIT_OR;
    // MODULE ADREESSES //
    localparam GPIO_BASEADDR = 16'h0000;
    localparam GPIO_HIGHADDR = 16'h1000-1;
    
    localparam SPI_BASEADDR = 16'h1000; //0x1000
    localparam SPI_HIGHADDR = 16'h2000-1;   //0x300f
    
    localparam FAST_SR_AQ_BASEADDR = 16'h2000;                    
    localparam FAST_SR_AQ_HIGHADDR = 16'h3000-1; 
 
    localparam PULSE_TRIGGER_BASEADDR = 16'h3000;                    
    localparam PULSE_TRIGGER_HIGHADDR = 16'h4000-1; 
    
    localparam PULSE_INJ_BASEADDR = 16'h4000;
    localparam PULSE_INJ_HIGHADDR = 16'h5000-1;  

    localparam PULSE_TESTHIT_BASEADDR = 16'h5000;
    localparam PULSE_TESTHIT_HIGHADDR = 16'h6000-1;  
    
    localparam FERX_BASEADDR = 16'h6000;
    localparam FERX_HIGHADDR = 16'h7000-1;  
    
    localparam FIFO_BASEADDR = 16'h8000;
    localparam FIFO_HIGHADDR = 16'h9000-1;
 
    //localparam TDC_BASEADDR = 16'h9000;
    //localparam TDC_HIGHADDR = 16'ha000-1;

    localparam TDC_BASEADDR = 16'h9000;
    localparam TDC_HIGHADDR = 16'h9100-1;
     
    localparam TLU_BASEADDR = 16'h9100;
    localparam TLU_HIGHADDR = 16'h9500-1;
     
    localparam PULSE_TLU_VETO_BASEADDR = 16'h9500;
    localparam PULSE_TLU_VETO_HIGHADDR = 16'h9a00-1;

    // ------- RESRT/CLOCK  ------- //

    (* KEEP = "{TRUE}" *) 
    wire CLK320;  
    (* KEEP = "{TRUE}" *) 
    wire CLK160;
    (* KEEP = "{TRUE}" *) 
    wire CLK40;
    (* KEEP = "{TRUE}" *) 
    wire CLK16;
    (* KEEP = "{TRUE}" *) 
    wire BUS_CLK;
    (* KEEP = "{TRUE}" *) 
    wire CLK8;
    (* KEEP = "{TRUE}" *) 
    wire CLK1;
    
    clock_divider #(
        .DIVISOR(8)
    ) i_clock_divisor_1MHz (
        .CLK(CLK8),
        .RESET(1'b0),
        .CE(),
        .CLOCK(CLK1)
    );
    
    reg HIT_OR_TDC_OUT;
    wire CLK_LOCKED;
    
    clk_gen iclk_gen(
        .CLKIN(FCLK_IN),
        .BUS_CLK(BUS_CLK),
        .U1_CLK8(CLK8),
        .U2_CLK40(CLK40),
        .U2_CLK16(CLK16),
        .U2_CLK160(CLK160),
        .U2_CLK320(CLK320),
        .U2_LOCKED(CLK_LOCKED)
    );

    wire BUS_RST;
    reset_gen ireset_gen(.CLK(BUS_CLK), .RST(BUS_RST));
    

    // -------  BUS SYGNALING  ------- //
    wire [15:0] BUS_ADD;
    wire BUS_RD, BUS_WR;
    fx2_to_bus i_fx2_to_bus (
        .ADD(ADD),
        .RD_B(RD_B),
        .WR_B(WR_B),

        .BUS_CLK(BUS_CLK),
        .BUS_ADD(BUS_ADD),
        .BUS_RD(BUS_RD),
        .BUS_WR(BUS_WR)
    );
    

    // ------- MODULES  ------- //
    wire [15:0] GPIO_OUT;
    gpio 
    #( 
        .BASEADDR(GPIO_BASEADDR), 
        .HIGHADDR(GPIO_HIGHADDR),
        .IO_WIDTH(16),
        .IO_DIRECTION(16'hffff)
    ) i_gpio
    (
        .BUS_CLK(BUS_CLK),
        .BUS_RST(BUS_RST),
        .BUS_ADD(BUS_ADD),
        .BUS_DATA(BUS_DATA[15:0]),
        .BUS_RD(BUS_RD),
        .BUS_WR(BUS_WR),
        .IO(GPIO_OUT)
    );    
    
    wire DISABLE_LD, LD;
    assign #1000 DUT_RESET = GPIO_OUT[1:0];
    ODDR clk_bx_gate(.D1(GPIO_OUT[2]), .D2(1'b0), .C(CLK40), .CE(1'b1), .R(1'b0), .S(1'b0), .Q(DUT_CLK_BX) );
    ODDR clk_out_gate(.D1(GPIO_OUT[6]), .D2(1'b0), .C(CLK160), .CE(1'b1), .R(1'b0), .S(1'b0), .Q(DUT_CLK_DATA) );
//  ODDR clk_data_gate(.D1(1'b1), .D2(1'b0), .C(CLK160), .CE(1'b1), .R(1'b0), .S(1'b0), .Q(LEMO_TX[0]) );
//  ODDR clk_pollo_gate(.D1(1'b1), .D2(1'b0), .C(CLK40), .CE(1'b1), .R(1'b0), .S(1'b0), .Q(LEMO_TX[1]) );
//  ODDR clk_cane_gate(.D1(1'b1), .D2(1'b0), .C(CLK8), .CE(1'b1), .R(1'b0), .S(1'b0), .Q(LEMO_TX[2]) );
    
    assign DUT_PIX_D_CONF = GPIO_OUT[3];
    wire GATE_EN_PIX_SR_CNFG;
    assign GATE_EN_PIX_SR_CNFG = GPIO_OUT[4];
    assign DISABLE_LD = GPIO_OUT[5];
    assign LD = GPIO_OUT[7];
    wire EXT_TRIGGER_ENABLE_CONF;
    assign EXT_TRIGGER_ENABLE_CONF = GPIO_OUT[9];
    
    //assign EN_HIT_OR_TRIGGER = GPIO_OUT[8];
    //assign EN_HIT_OR_TRIGGER = 1;
    
    wire SCLK, SDI, SDO, SEN, SLD;
    
    spi 
    #( 
        .BASEADDR(SPI_BASEADDR), 
        .HIGHADDR(SPI_HIGHADDR),
        .MEM_BYTES(512) 
    )  i_spi
    (
        .BUS_CLK(BUS_CLK),
        .BUS_RST(BUS_RST),
        .BUS_ADD(BUS_ADD),
        .BUS_DATA(BUS_DATA[7:0]),
        .BUS_RD(BUS_RD),
        .BUS_WR(BUS_WR),
    
        .SPI_CLK(CLK8),
    
        .SCLK(SCLK),
        .SDI(SDI),
        .SDO(SDO),
        .SEN(SEN),
        .SLD(SLD)
    );
    
    reg [2:0] delay_cnt;
    always@(posedge CLK1)
        if(BUS_RST)
            delay_cnt <= 0;
        else if(SEN)
            delay_cnt <= 3'b111;
        else if(delay_cnt != 0)
            delay_cnt <= delay_cnt - 1;
         
    wire TESTHIT;
    
    assign SDO = DUT_SO_CNFG;
    assign DUT_SI_CNFG = SDI;    
    assign DUT_CLK_CNFG = SCLK; //~
    assign DUT_EN_PIX_SR_CNFG = /*(SEN| (|delay_cnt) ) &*/ GATE_EN_PIX_SR_CNFG; 
    assign DUT_LD_CNFG = (SLD & !DISABLE_LD) | TESTHIT | LD;

    pulse_gen
    #( 
        .BASEADDR(PULSE_INJ_BASEADDR), 
        .HIGHADDR(PULSE_INJ_HIGHADDR)
    ) i_pulse_gen_inj
    (
        .BUS_CLK(BUS_CLK),
        .BUS_RST(BUS_RST),
        .BUS_ADD(BUS_ADD),
        .BUS_DATA(BUS_DATA[7:0]),
        .BUS_RD(BUS_RD),
        .BUS_WR(BUS_WR),
    
        .PULSE_CLK(~CLK40),
        .EXT_START(SLD),
        .PULSE(DUT_INJ) //DUT_INJ
    );
    
    pulse_gen
    #( 
        .BASEADDR(PULSE_TESTHIT_BASEADDR), 
        .HIGHADDR(PULSE_TESTHIT_HIGHADDR)
    ) i_pulse_gen_testhit
    (
        .BUS_CLK(BUS_CLK),
        .BUS_RST(BUS_RST),
        .BUS_ADD(BUS_ADD),
        .BUS_DATA(BUS_DATA[7:0]),
        .BUS_RD(BUS_RD),
        .BUS_WR(BUS_WR),
    
        .PULSE_CLK(~CLK40),
        .EXT_START(SLD),
        .PULSE(TESTHIT)
    );

    wire TDC_OUT;
    wire TRIG_READY_FLAG, TRIGGER_ACCEPTED_FLAG, TRIG_READY_OUT;
    reg TRIG_READY_OUT_FF;

    pulse_gen
    #( 
        .BASEADDR(PULSE_TRIGGER_BASEADDR), 
        .HIGHADDR(PULSE_TRIGGER_HIGHADDR)
    ) i_pulse_gen_trigger
    (
        .BUS_CLK(BUS_CLK),
        .BUS_RST(BUS_RST),
        .BUS_ADD(BUS_ADD),
        .BUS_DATA(BUS_DATA[7:0]),
        .BUS_RD(BUS_RD),
        .BUS_WR(BUS_WR),
    
        .PULSE_CLK(~CLK40),
        .EXT_START(HIT_OR_TDC_OUT),
        //.EXT_START( (TESTHIT | DUT_INJ | TDC_OUT) & TRIGGER_ACCEPTED_FLAG),
        .PULSE(DUT_TRIGGER),
	    .READY(TRIG_READY_OUT)
    );

    always @ (posedge CLK40) 
    begin
        TRIG_READY_OUT_FF <= TRIG_READY_OUT;
    end
    assign TRIG_READY_FLAG = ~TRIG_READY_OUT & TRIG_READY_OUT_FF;

    wire TLU_FIFO_READ, TLU_FIFO_EMPTY, TLU_FIFO_PEEMPT_REQ;
    wire [31:0] TLU_FIFO_DATA;
    wire FIFO_FULL, TLU_TRIG_EN;
    tlu_controller #(
        .BASEADDR(TLU_BASEADDR),
        .HIGHADDR(TLU_HIGHADDR),
        .TLU_TRIGGER_MAX_CLOCK_CYCLES(32),
        .DIVISOR(8)
    ) i_tlu_controller (
        .BUS_CLK(BUS_CLK),
        .BUS_RST(BUS_RST),
        .BUS_ADD(BUS_ADD),
        .BUS_DATA(BUS_DATA),
        .BUS_RD(BUS_RD),
        .BUS_WR(BUS_WR),
         
        .TRIGGER_CLK(~CLK40),
         
        .FIFO_READ(TLU_FIFO_READ),
        .FIFO_EMPTY(TLU_FIFO_EMPTY),
        .FIFO_DATA(TLU_FIFO_DATA),
         
        .FIFO_PREEMPT_REQ(TLU_FIFO_PEEMPT_REQ),
         
        //.TRIGGER({8'b0}),
        .TRIGGER({7'b0, TDC_OUT}),
        .TRIGGER_VETO({7'b0, FIFO_FULL}),
        .TRIGGER_ENABLED(TLU_TRIG_EN),
         
        .TRIGGER_ACKNOWLEDGE(TRIG_READY_OUT),
        .TRIGGER_ACCEPTED_FLAG(TRIGGER_ACCEPTED_FLAG),
        .EXT_TRIGGER_ENABLE(1'b1),

        .TLU_TRIGGER(1'b0),
        .TLU_RESET(1'b0),
        .TLU_BUSY(),
        .TLU_CLOCK()
         
        //.TIMESTAMP(16'b0)
    );

    wire ARB_READY_OUT, ARB_WRITE_OUT;
    wire [31:0] ARB_DATA_OUT;
    
    wire FE_FIFO_READ;
    wire FE_FIFO_EMPTY;
    wire [31:0] FE_FIFO_DATA;
    
    wire TDC_FIFO_READ;
    wire TDC_FIFO_EMPTY;
    wire [31:0] TDC_FIFO_DATA;

    rrp_arbiter 
    #( 
        .WIDTH(3)
    ) i_rrp_arbiter
    (
        .RST(BUS_RST),
        .CLK(BUS_CLK),
    
        .WRITE_REQ({~FE_FIFO_EMPTY, ~TDC_FIFO_EMPTY, ~TLU_FIFO_EMPTY}),
        .HOLD_REQ({2'b0, TLU_FIFO_PEEMPT_REQ}),
        .DATA_IN({FE_FIFO_DATA, TDC_FIFO_DATA, TLU_FIFO_DATA}),
        .READ_GRANT({FE_FIFO_READ, TDC_FIFO_READ, TLU_FIFO_READ}),

        .READY_OUT(ARB_READY_OUT),
        .WRITE_OUT(ARB_WRITE_OUT),
        .DATA_OUT(ARB_DATA_OUT)
    );
    
    wire RX_READY, RX_8B10B_DECODER_ERR, RX_FIFO_OVERFLOW_ERR, RX_FIFO_FULL;
    fei4_rx #(
    .BASEADDR(FERX_BASEADDR),
    .HIGHADDR(FERX_HIGHADDR),
    .DSIZE(10),
    .DATA_IDENTIFIER(0)
    ) i_fei4_rx (
        .RX_CLK(CLK160), //CLK160
        .RX_CLK2X(CLK320), //CLK320
        .DATA_CLK(CLK16), //CLK16

        .RX_DATA(DUT_OUT_DATA),

        .RX_READY(RX_READY),
        .RX_8B10B_DECODER_ERR(RX_8B10B_DECODER_ERR),
        .RX_FIFO_OVERFLOW_ERR(RX_FIFO_OVERFLOW_ERR),

        .FIFO_READ(FE_FIFO_READ),
        .FIFO_EMPTY(FE_FIFO_EMPTY),
        .FIFO_DATA(FE_FIFO_DATA),

        .RX_FIFO_FULL(RX_FIFO_FULL),

        .BUS_CLK(BUS_CLK),
        .BUS_RST(BUS_RST),
        .BUS_ADD(BUS_ADD),
        .BUS_DATA(BUS_DATA),
        .BUS_RD(BUS_RD),
        .BUS_WR(BUS_WR)
    );

    tdc_s3 #(
        .BASEADDR(TDC_BASEADDR),
        .HIGHADDR(TDC_HIGHADDR),
        .CLKDV(4),
        .DATA_IDENTIFIER(4'b0100), 
        .FAST_TDC(1),
        .FAST_TRIGGER(0)
    ) i_tdc (
        .CLK320(CLK320),
        .CLK160(CLK160),
        .DV_CLK(CLK40),
        .TDC_IN(DUT_HIT_OR),
        .TDC_OUT(TDC_OUT),
        .TRIG_IN(1'b0),
        .TRIG_OUT(),

        .FIFO_READ(TDC_FIFO_READ),
        .FIFO_EMPTY(TDC_FIFO_EMPTY),
        .FIFO_DATA(TDC_FIFO_DATA),

        .BUS_CLK(BUS_CLK),
        .BUS_RST(BUS_RST),
        .BUS_ADD(BUS_ADD),
        .BUS_DATA(BUS_DATA),
        .BUS_RD(BUS_RD),
        .BUS_WR(BUS_WR),

        .ARM_TDC(1'b0),
        .EXT_EN(TLU_TRIG_EN),
        
        .TIMESTAMP(16'b0)
    );

    always@(negedge CLK40)
		HIT_OR_TDC_OUT <= TDC_OUT;
    assign LEMO_TX[0] = TRIGGER_ACCEPTED_FLAG;
	assign LEMO_TX[1] = TRIG_READY_FLAG;
	assign LEMO_TX[2] = 1'b0;
	 
    wire USB_READ;
    assign USB_READ = FREAD & FSTROBE;
    
    
    sram_fifo #(
        .BASEADDR(FIFO_BASEADDR),
        .HIGHADDR(FIFO_HIGHADDR)
    ) i_out_fifo (
        .BUS_CLK(BUS_CLK),
        .BUS_RST(BUS_RST),
        .BUS_ADD(BUS_ADD),
        .BUS_DATA(BUS_DATA),
        .BUS_RD(BUS_RD),
        .BUS_WR(BUS_WR), 

        .SRAM_A(SRAM_A),
        .SRAM_IO(SRAM_IO),
        .SRAM_BHE_B(SRAM_BHE_B),
        .SRAM_BLE_B(SRAM_BLE_B),
        .SRAM_CE1_B(SRAM_CE1_B),
        .SRAM_OE_B(SRAM_OE_B),
        .SRAM_WE_B(SRAM_WE_B),

        .USB_READ(USB_READ),
        .USB_DATA(FDATA),

        .FIFO_READ_NEXT_OUT(ARB_READY_OUT),
        .FIFO_EMPTY_IN(!ARB_WRITE_OUT),
        .FIFO_DATA(ARB_DATA_OUT),

        .FIFO_NOT_EMPTY(),
        .FIFO_FULL(FIFO_FULL),
        .FIFO_NEAR_FULL(),
        .FIFO_READ_ERROR()
    );
    
    assign SDA = 1'bz;
    assign SCL = 1'bz;

    wire CLK_3HZ;
    clock_divider #(
        .DIVISOR(13333333)
    ) i_clock_divisor_40MHz_to_3Hz (
        .CLK(CLK40),
        .RESET(1'b0),
        .CE(),
        .CLOCK(CLK_3HZ)
    );

    wire CLK_1HZ;
    clock_divider #(
        .DIVISOR(40000000)
    ) i_clock_divisor_40MHz_to_1Hz (
        .CLK(CLK40),
        .RESET(1'b0),
        .CE(),
        .CLOCK(CLK_1HZ)
    );
    
    assign LED[2:0] = 3'b000;
    assign LED[3] = RX_READY & ((RX_8B10B_DECODER_ERR? CLK_3HZ : CLK_1HZ) | RX_FIFO_OVERFLOW_ERR | RX_FIFO_FULL);
    assign LED[4] = (CLK_1HZ | FIFO_FULL) & CLK_LOCKED;
    
    
    
    /*
    wire [35:0] control_bus;
    chipscope_icon ichipscope_icon
    (
        .CONTROL0(control_bus)
    ); 

    chipscope_ila ichipscope_ila 
    (
        .CONTROL(control_bus),
        .CLK(CLK80),
        .TRIG0({DUT_HIT_OR, DUT_SO_CNFG , DUT_OUT_DATA}) //
    ); 
    */
    
endmodule
