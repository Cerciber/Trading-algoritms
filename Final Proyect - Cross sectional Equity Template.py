# Importar librerias necesarias
import quantopian.algorithm as algo
import quantopian.optimize as opt 
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import SimpleMovingAverage
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.experimental import risk_loading_pipeline
from quantopian.pipeline.data.psychsignal import stocktwits
from quantopian.pipeline.data import Fundamentals
import quantopian.pipeline.data.factset.estimates as fe
from quantopian.pipeline.data import EquityPricing

# Parametros de restricciones
MAX_GROSS_LEVERAGE = 1.1
TOTAL_POSITIONS = 100
  
# Limites maximo y minimo del dinero que se invierte en cada accion (Donde 1 es el 100%)
MAX_SHORT_POSITION_SIZE = 1.0 / TOTAL_POSITIONS
MAX_LONG_POSITION_SIZE = 1.0 / TOTAL_POSITIONS

# Inicializar (Se llama exactamente una vez cuando el algoritmo comienza a ejecutarse)
def initialize(context):

    # Agregar tuberia 
    algo.attach_pipeline(make_pipeline(), 'long_short_equity_template')

    # Agregar tuberia de riesgo    
    algo.attach_pipeline(risk_loading_pipeline(), 'risk_factors')

    # Agregar función programada de rebalanceo
    algo.schedule_function(func=rebalance,
                           date_rule=algo.date_rules.week_start(),
                           time_rule=algo.time_rules.market_open(hours=0, minutes=30),
                           half_days=True)

    # Agregar función programada de registro de variables
    algo.schedule_function(func=record_vars,
                           date_rule=algo.date_rules.every_day(),
                           time_rule=algo.time_rules.market_close(),
                           half_days=True)

# Tuberia
def make_pipeline():
    
    # Ganancias por acción con frecuencia semestral reportada mas recientemente
    profits = fe.Actuals.slice('EPS', 'qf', 0).actual_value.latest
    
    # Estimación de ganancias por acción con frecuencia trimentral
    profits_estimation = fe.PeriodicConsensus.slice('EPS', 'qf', 0).mean.latest
    
    # Diferencia entre el valor real y el estimado
    difference_profits = (profits - profits_estimation) / profits_estimation
    
    # Flujo de caja por acción con frecuencia semestral reportada mas recientemente
    flow = fe.Actuals.slice('CFPS', 'qf', 0).actual_value.latest
    
    # Flujo de caja por acción con frecuencia trimentral
    flow_estimation = fe.PeriodicConsensus.slice('CFPS', 'qf', 0).mean.latest
    
    # Diferencia entre el valor real y el estimado
    difference_flow = (flow - flow_estimation) / flow_estimation
    
    # Precio de cierre
    close = EquityPricing.close.latest
    
    # Precio de apertura
    open = EquityPricing.open.latest
    
    # Diferencia entre el precio de apretura y de cierre 
    difference_open_close_price = (close - open) / open
    
    # Precio mas alto
    high = EquityPricing.high.latest
    
    # Precio mas bajo
    low = EquityPricing.low.latest
    
    # diferencia entre el precio mas alto y el mas bajo
    difference_high_low_price = (high - low) / low
    
    # N° acciones negociadas
    volume = EquityPricing.volume.latest
    
    # factores basado en datos de fundamentals y SimpleMovingAverage
    value = Fundamentals.ebit.latest / Fundamentals.market_cap.latest
    quality = Fundamentals.roe.latest
    sentiment_score = SimpleMovingAverage(
        inputs=[stocktwits.bull_minus_bear],
        window_length=3,
    )

    # Obtener universo de acciones negociables
    universe = QTradableStocksUS()
    
    # Cambiar los valores por fuera de los percentiles minimo y maximo por el valor del percentil del extremo correspondiente        
    difference_profits_winsorized = -difference_profits.winsorize(min_percentile=0.05, max_percentile=0.95)
    difference_flow_winsorized =  -difference_flow.winsorize(min_percentile=0.05, max_percentile=0.95)
    difference_open_close_price_winsorized = -difference_open_close_price.winsorize(min_percentile=0.05, max_percentile=0.95)
    difference_high_low_price_winsorized = -difference_high_low_price.winsorize(min_percentile=0.05, max_percentile=0.95)
    volume_winsorized = -volume.winsorize(min_percentile=0.05, max_percentile=0.95)
    value_winsorized = -value.winsorize(min_percentile=0.05, max_percentile=0.95)
    quality_winsorized = -quality.winsorize(min_percentile=0.05, max_percentile=0.95)
    sentiment_score_winsorized = -sentiment_score.winsorize(min_percentile=0.05,                                                                             max_percentile=0.95)

    # Normalizar los datos y sumarlos para amplificar
    combined_factor = ( 
        difference_profits_winsorized.zscore() +
        difference_flow_winsorized.zscore() + 
        difference_open_close_price_winsorized.zscore() + 
        difference_high_low_price_winsorized.zscore() + 
        volume_winsorized.zscore() + 
        value_winsorized.zscore() + 
        quality_winsorized.zscore() + 
        sentiment_score_winsorized.zscore()
    )

    # Filtrar los primeros y los ultimos valores
    longs = combined_factor.top(TOTAL_POSITIONS//2, mask=universe)
    shorts = combined_factor.bottom(TOTAL_POSITIONS//2, mask=universe)

    # Unir los primeros y los ultimos valores
    long_short_screen = (longs | shorts)

    # Crear tuberia
    pipe = Pipeline(
        columns={
            'longs': longs,
            'shorts': shorts,
            'combined_factor': combined_factor
        },
        screen=long_short_screen
    )
    return pipe
 
# Antes de iniciar el comercio (Se llama una vez al día antes de que abra el mercado)
def before_trading_start(context, data):
    
    # Guardar dataFrame de la tuberia
    context.pipeline_data = algo.pipeline_output('long_short_equity_template')

    # Guardar dataFrame de la tuberia de riesgo
    context.risk_loadings = algo.pipeline_output('risk_factors')

# Registro de variables
def record_vars(context, data):
    
    # Graficar el numero de posiciones en el tiempo
    algo.record(num_positions=len(context.portfolio.positions))


# Función de rebalanceo
def rebalance(context, data):
    
    # Obtener dataframes
    pipeline_data = context.pipeline_data
    risk_loadings = context.risk_loadings

    # Objetivo: Maximizar retornos para el factor combinado
    objective = opt.MaximizeAlpha(pipeline_data.combined_factor)

    # Definir la lista de restricciones
    constraints = []
    
    # Restricción: Limitar la exposición maxima de la inversión (Donde 1 es el 100%)
    constraints.append(opt.MaxGrossExposure(MAX_GROSS_LEVERAGE))

    # Restricción: Realizar Long y short en la misma proporción
    constraints.append(opt.DollarNeutral())

    # Crear restricción del modelo de riesgo
    neutralize_risk_factors = opt.experimental.RiskModelExposure(
        risk_model_loadings=risk_loadings,
        version=0
    )
    constraints.append(neutralize_risk_factors)

    # Restricción: Limitar la cantidad de dinero que se invierte en una sola acción (Donde 1 es el 100%)
    constraints.append(
        opt.PositionConcentration.with_equal_bounds(
            min=-MAX_SHORT_POSITION_SIZE,
            max=MAX_LONG_POSITION_SIZE
        ))

    # Calcular pesos de una cartera óptima y haga pedidos hacia esa cartera segun el objetivo y las restricciones especificadas
    algo.order_optimal_portfolio(
        objective=objective,
        constraints=constraints
    )